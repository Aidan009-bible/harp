# harp_hand_detector.py  v5 — Orientation-Agnostic String Detection
# Works for any string angle: horizontal, vertical, diagonal.
# Each string is a line segment (two endpoints) derived from YOLO bbox.
# Touch = fingertip closest to a string centerline within threshold.
# pip install ultralytics mediapipe==0.10.9 opencv-python numpy

import cv2, os, time, csv, argparse, json
import numpy as np
from ultralytics import YOLO
try:
    # MediaPipe 0.10.30+ uses Tasks API
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import Image as MPImage, ImageFormat
    MP_NEW_API = True
except ImportError:
    try:
        # Fallback to old solutions API
        import mediapipe as mp
        MP_NEW_API = False
    except ImportError as e:
        raise ImportError(f"MediaPipe not installed: {e}")
from collections import defaultdict

# ============================= CONFIG =============================
WEIGHTS       = "best.pt"
IMGSZ         = 960
CONF          = 0.08
IOU_THRESH    = 0.40
NUM_STRINGS   = 16
MODEL_HISTORY = 80
TOUCH_DIST_PX = 20             # max distance from fingertip to string
TOUCH_CONSEC  = 1              # instant detection (plucks are 1-2 frames)
OUTPUT_DIR    = "output"
CSV_LOG       = os.path.join(OUTPUT_DIR, "touch_events.csv")
# ==================================================================

FINGER_TIPS = {4: "thumb", 8: "index"}

HAND_BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

def _scol(i, n=16):
    hue = int(120 * i / (n - 1))
    hsv = np.uint8([[[hue, 180, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

SCOLS = [_scol(i) for i in range(NUM_STRINGS)]

# =============================== NMS ================================

def nms(boxes, scores, iou_thr=0.40):
    if len(boxes) == 0:
        return []
    b = boxes.astype(np.float32)
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    area = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1+1); h = np.maximum(0, yy2-yy1+1)
        iou = (w*h)/(area[i]+area[order[1:]]-w*h+1e-9)
        order = order[np.where(iou <= iou_thr)[0]+1]
    return keep

def extract_boxes(result, conf_thr, iou_thr):
    out = []
    if not hasattr(result, "boxes") or len(result.boxes) == 0:
        return out
    xyxy  = np.array(result.boxes.xyxy.tolist(), np.float32)
    cls   = np.array(result.boxes.cls.tolist(),  np.int32)
    confs = np.array(result.boxes.conf.tolist(), np.float32)
    ok = confs >= conf_thr
    if not ok.any(): return out
    xyxy, cls, confs = xyxy[ok], cls[ok], confs[ok]
    for c in np.unique(cls):
        m = cls == c
        keep = nms(xyxy[m], confs[m], iou_thr)
        for k in keep:
            idx = np.where(m)[0][k]
            out.append(dict(
                xyxy=tuple(map(float, xyxy[idx])),
                cid=int(c), conf=float(confs[idx])))
    return out

# ============= GEOMETRY: orientation-agnostic =============

def bbox_to_centerline(x1, y1, x2, y2):
    """
    Convert a bounding box into a centerline segment (two endpoints).
    The centerline runs along the LONG axis of the bbox.
    Works for horizontal, vertical, and diagonal strings.
    """
    w, h = x2 - x1, y2 - y1
    if w >= h:
        # Wider than tall → string runs mostly left-to-right
        cy = (y1 + y2) / 2.0
        return (x1, cy), (x2, cy)
    else:
        # Taller than wide → string runs mostly top-to-bottom
        cx = (x1 + x2) / 2.0
        return (cx, y1), (cx, y2)


def point_to_segment_dist(px, py, ax, ay, bx, by):
    """
    Exact distance from point (px,py) to line segment (ax,ay)→(bx,by).
    Returns (distance, snap_x, snap_y) where snap is the closest point
    on the segment.
    """
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    d2 = vx*vx + vy*vy
    if d2 < 1e-9:
        d = float(np.hypot(px - ax, py - ay))
        return d, ax, ay
    t = max(0.0, min(1.0, (vx*wx + vy*wy) / d2))
    sx = ax + t * vx
    sy = ay + t * vy
    return float(np.hypot(px - sx, py - sy)), sx, sy

# ============= STRING MODEL: stores centerline endpoints =============

class StringModel:
    """
    Orientation-agnostic string model.
    Each string stored as a centerline: two endpoints (ax,ay) → (bx,by).
    Accumulates observations over time, fits polynomial for interpolation.
    """
    def __init__(self, n=NUM_STRINGS, history=MODEL_HISTORY):
        self.n = n
        self.history = history
        self.obs = defaultdict(list)   # cid → [(ax,ay,bx,by,conf), ...]
        self.ready = False
        self.lines = None              # list of ((ax,ay),(bx,by)) per string
        self.n_seen = 0

    def feed(self, boxes):
        for b in boxes:
            cid = b["cid"]
            if not (0 <= cid < self.n):
                continue
            x1, y1, x2, y2 = b["xyxy"]
            (ax, ay), (bx, by) = bbox_to_centerline(x1, y1, x2, y2)
            self.obs[cid].append((ax, ay, bx, by, b["conf"]))
            if len(self.obs[cid]) > self.history:
                self.obs[cid] = self.obs[cid][-self.history:]
        self._fit()

    def _wavg(self, vals_weights):
        r = vals_weights[-25:]
        tw = sum(w for _, w in r)
        return sum(v*w for v, w in r) / tw if tw > 1e-9 else None

    def _fit(self):
        detected = {}
        for cid, obs in self.obs.items():
            if len(obs) < 2:
                continue
            ax = self._wavg([(o[0], o[4]) for o in obs])
            ay = self._wavg([(o[1], o[4]) for o in obs])
            bx = self._wavg([(o[2], o[4]) for o in obs])
            by = self._wavg([(o[3], o[4]) for o in obs])
            if ax is not None:
                detected[cid] = (ax, ay, bx, by)

        self.n_seen = len(detected)
        if len(detected) < 3:
            self.ready = False
            return

        ids = sorted(detected.keys())
        axs = np.array([detected[i][0] for i in ids])
        ays = np.array([detected[i][1] for i in ids])
        bxs = np.array([detected[i][2] for i in ids])
        bys = np.array([detected[i][3] for i in ids])

        deg = min(2, len(ids) - 1)
        try:
            pa = np.polyfit(ids, axs, deg)
            pb = np.polyfit(ids, ays, deg)
            pc = np.polyfit(ids, bxs, deg)
            pd = np.polyfit(ids, bys, deg)
        except np.linalg.LinAlgError:
            self.ready = False
            return

        self.lines = []
        for i in range(self.n):
            if i in detected:
                d = detected[i]
                self.lines.append(((d[0], d[1]), (d[2], d[3])))
            else:
                self.lines.append((
                    (float(np.polyval(pa, i)), float(np.polyval(pb, i))),
                    (float(np.polyval(pc, i)), float(np.polyval(pd, i)))))
        self.ready = True

# ============= TOUCH DETECTION: closest string per fingertip =============

def detect_touches(fingertips, model, yolo_boxes, dist_thr):
    """
    For each fingertip, find the SINGLE closest string (from model or
    YOLO boxes). This prevents one finger from "touching" multiple strings.
    Returns list of (finger_name, lm_idx, sid, sname, dist, snap_x, snap_y).
    """
    results = []

    for lm_idx, fname, px, py in fingertips:
        best_sid  = -1
        best_dist = dist_thr + 1
        best_snap = (0, 0)

        # Check geometric model
        if model.ready:
            for sid, ((ax, ay), (bx, by)) in enumerate(model.lines):
                d, sx, sy = point_to_segment_dist(px, py, ax, ay, bx, by)
                if d < best_dist:
                    best_dist, best_sid, best_snap = d, sid, (sx, sy)

        # Check raw YOLO boxes (may be more accurate this frame)
        for b in yolo_boxes:
            sid = b["cid"]
            x1, y1, x2, y2 = b["xyxy"]
            (ax, ay), (bx, by) = bbox_to_centerline(x1, y1, x2, y2)
            d, sx, sy = point_to_segment_dist(px, py, ax, ay, bx, by)
            if d < best_dist:
                best_dist, best_sid, best_snap = d, sid, (sx, sy)

        if best_sid >= 0 and best_dist <= dist_thr:
            results.append((fname, lm_idx, best_sid, f"S{best_sid+1}",
                            best_dist, best_snap[0], best_snap[1]))

    return results

# ============================== DRAWING ==============================

def draw_strings(frame, model, yolo_boxes, touched_ids, contacts, fidx):
    overlay = frame.copy()

    if model.ready:
        for i, ((ax, ay), (bx, by)) in enumerate(model.lines):
            a = (int(ax), int(ay))
            b = (int(bx), int(by))
            col = SCOLS[i]
            touched = i in touched_ids

            if touched:
                cv2.line(overlay, a, b, (0, 255, 255), 10, cv2.LINE_AA)
                cv2.line(frame,   a, b, (255, 255, 255), 3, cv2.LINE_AA)
            else:
                cv2.line(frame, a, b, col, 1, cv2.LINE_AA)

            # Label near left endpoint
            lx, ly = min(a[0], b[0]) - 36, (a[1] + b[1]) // 2 + 4
            lx = max(4, lx)
            label = f"S{i+1}"
            if touched:
                cv2.putText(frame, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            col, 1, cv2.LINE_AA)
    else:
        # Warmup: show raw YOLO centerlines
        for b in yolo_boxes:
            x1, y1, x2, y2 = b["xyxy"]
            (ax, ay), (bx, by) = bbox_to_centerline(x1, y1, x2, y2)
            a, b2 = (int(ax), int(ay)), (int(bx), int(by))
            cv2.line(frame, a, b2, (80, 80, 80), 1)
            cv2.putText(frame, f"S{b['cid']+1}",
                        (max(4, a[0]-36), a[1]+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120,120,120), 1)

    # Blend glow
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    # Contact connectors
    pulse = int(6 + 3 * abs(np.sin(fidx * 0.4)))
    for fpx, fpy, sx, sy, sid in contacts:
        fi = (int(fpx), int(fpy))
        si = (int(sx), int(sy))
        cv2.line(frame, fi, si, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, si, pulse, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, si, 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_hand(frame, landmarks, W, H):
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in landmarks.landmark]
    for a, b in HAND_BONES:
        cv2.line(frame, pts[a], pts[b], (50, 200, 100), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        if i in FINGER_TIPS:
            cv2.circle(frame, (x, y), 9, (255, 50, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 9, (255, 255, 255), 2, cv2.LINE_AA)
            lbl = FINGER_TIPS[i][:3].upper()
            cv2.putText(frame, lbl, (x+11, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 2, (60, 160, 60), -1, cv2.LINE_AA)
    return pts


def draw_subtitle(frame, touches, W, H):
    bar_h = 48
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H-bar_h), (W, H), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    if not touches:
        cv2.putText(frame, "---", (W//2 - 20, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (80, 80, 80), 1, cv2.LINE_AA)
        return

    seen = set()
    parts = []
    for fn, _, sid, sn, dist, _, _ in touches:
        key = (sn, fn)
        if key not in seen:
            seen.add(key)
            parts.append(f"{sn} ({fn})")
    text = "  |  ".join(parts)
    tsz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
    tx = max(10, (W - tsz[0]) // 2)
    cv2.putText(frame, text, (tx, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (0, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, model, n_hands, fps_p, fidx, total):
    if model.ready:
        st = f"READY ({model.n_seen}/{NUM_STRINGS} seen)"
        col = (100, 220, 100)
    else:
        st = "LEARNING..."
        col = (100, 100, 200)
    txt = f"{st}  |  Hands: {n_hands}  |  {fps_p:.0f} fps"
    cv2.putText(frame, txt, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
    if total > 0:
        pct = fidx / total
        cv2.rectangle(frame, (10, 28), (170, 34), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 28), (10+int(160*pct), 34), (0, 200, 0), -1)

# ============================== CSV ==============================

def log_touch(ev, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "frame", "finger", "string", "sid", "dist_px"])
        w.writerow([ev["ts"], ev["frame"], ev["finger"],
                     ev["string"], ev["sid"], f"{ev['dist']:.1f}"])

# ============================== MAIN ==============================

def run(source, out_video=None, preview=True, output_dir=None, weights_path=None, fast_mode=False, audio_onsets=None):
    global CSV_LOG
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        CSV_LOG = os.path.join(output_dir, "touch_events.csv")
        if out_video is None and not fast_mode:
            out_video = os.path.join(output_dir, "video_detected.mp4")
        preview = False
    if weights_path is None:
        weights_path = WEIGHTS

    print("=" * 64)
    print("  HARP TOUCH DETECTOR  v5")
    print("  Orientation-Agnostic | Closest-String-Per-Finger")
    print("=" * 64)
    print(f"  Weights    : {weights_path}  imgsz={IMGSZ}  conf>={CONF}")
    print(f"  Touch dist : {TOUCH_DIST_PX}px  consec={TOUCH_CONSEC}")
    print(f"  Fingers    : {list(FINGER_TIPS.values())}")
    print()

    yolo = YOLO(weights_path)
    names = yolo.names
    print(f"  Classes ({len(names)}): {names}")

    if MP_NEW_API:
        # MediaPipe 0.10.30+ Tasks API
        # Download model if needed, or use bundled model path
        import urllib.request
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            # Try to use MediaPipe's bundled model location
            try:
                import mediapipe.tasks.python.vision.hand_landmarker as hl_module
                # MediaPipe bundles the model, try to find it
                mp_dir = os.path.dirname(hl_module.__file__)
                bundled_model = os.path.join(mp_dir, "hand_landmarker.task")
                if os.path.exists(bundled_model):
                    model_path = bundled_model
                else:
                    # Download from MediaPipe
                    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    print(f"Downloading hand landmarker model...")
                    urllib.request.urlretrieve(model_url, model_path)
                    print(f"Downloaded to {model_path}")
            except Exception as e:
                print(f"Warning: Could not find/download model: {e}")
                model_path = None
        
        if model_path and os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
        else:
            # Try without model (may use bundled)
            base_options = python.BaseOptions()
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.40,
            min_tracking_confidence=0.40,
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(options)
    else:
        # Old solutions API
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.40,
            model_complexity=1,
        )

    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Determine stride for fast mode
    stride = 1
    if fast_mode:
        if fps >= 55:
            stride = 4 # 60fps -> 15fps
        elif fps >= 25:
            stride = 2 # 30fps -> 15fps
            
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"  Video      : {W}x{H} @ {fps:.1f} fps, {total} frames (stride={stride})")

    writer = None
    if not fast_mode and out_video:
        if out_video is None:
            stem = (os.path.splitext(os.path.basename(source))[0]
                    if source != "0" else "webcam")
            out_video = os.path.join(output_dir or OUTPUT_DIR, stem + "_detected.mp4")
        os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
        writer = cv2.VideoWriter(out_video,
                                  cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        print(f"  Output     : {out_video}\n")

    smodel      = StringModel()
    contact_ctr = defaultdict(int)
    fidx        = 0
    t0          = time.time()

    json_results = []
    
    try:
        yolo_interval = 1 if not smodel.ready else 30
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fidx += 1
            
            # Skip frames if striding
            if fidx % stride != 0 and fidx != 1:
                continue
                
            tnow = fidx / fps
            ts = f"{int(tnow//60):02d}:{tnow%60:05.2f}"

            # Check audio onsets
            is_active_frame = True
            if audio_onsets:
                # Active if within 0.1s of any audio onset
                is_active_frame = any(abs(tnow - onset_t) <= 0.1 for onset_t in audio_onsets)

            # 1) YOLO → feed model (only run if model is not ready, or every yolo_interval frames)
            if fidx <= 50 or fidx % yolo_interval == 0:
                res = yolo.predict(source=frame, imgsz=IMGSZ, conf=CONF,
                                   verbose=False, save=False)[0]
                boxes = extract_boxes(res, CONF, IOU_THRESH)
                smodel.feed(boxes)
                # Recalculate interval based on model readiness
                yolo_interval = 2 if not smodel.ready else 30
            
            # Skip MediaPipe and touch detection if not an active frame
            if not is_active_frame:
                continue

            # 2) MediaPipe hands
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if MP_NEW_API:
                # New Tasks API (MediaPipe 0.10.30+)
                mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
                detection_result = hand_landmarker.detect(mp_image)
                all_pts = []
                if detection_result.hand_landmarks:
                    for hlm in detection_result.hand_landmarks:
                        # Create wrapper: hlm is list of NormalizedLandmark, draw_hand expects .landmark attribute
                        hlm_wrapper = type('obj', (object,), {'landmark': hlm})()
                        pts = draw_hand(frame, hlm_wrapper, W, H)
                        all_pts.append(pts)
            else:
                # Old solutions API
                mp_res = hands.process(rgb)
                all_pts = []
                if mp_res.multi_hand_landmarks:
                    for hlm in mp_res.multi_hand_landmarks:
                        pts = draw_hand(frame, hlm, W, H)
                        all_pts.append(pts)

            # 3) Build fingertip list
            fingertips = []
            for hi, pts in enumerate(all_pts):
                for lm_idx, fname in FINGER_TIPS.items():
                    fingertips.append((lm_idx, fname, pts[lm_idx][0],
                                      pts[lm_idx][1]))

            # 4) Touch detection — closest string per fingertip
            raw = detect_touches(fingertips, smodel, boxes, TOUCH_DIST_PX)

            confirmed = []
            touched_ids = set()
            contacts = []
            frame_keys = set()

            for fn, li, sid, sn, dist, sx, sy in raw:
                key = (li, sid)
                frame_keys.add(key)
                contact_ctr[key] += 1
                if contact_ctr[key] >= TOUCH_CONSEC:
                    confirmed.append((fn, li, sid, sn, dist, sx, sy))
                    touched_ids.add(sid)
                    for _, _, px, py in fingertips:
                        # find matching fingertip
                        pass
                    # use the fingertip that triggered this
                    for _li, _fn, _px, _py in fingertips:
                        if _li == li:
                            contacts.append((_px, _py, sx, sy, sid))
                            break

            for key in list(contact_ctr):
                if key not in frame_keys:
                    contact_ctr[key] = max(0, contact_ctr[key] - 1)
                    if contact_ctr[key] == 0:
                        del contact_ctr[key]

            # Calculate FPS
            el = time.time() - t0
            fps_p = fidx / el if el > 0 else 0

            # 5) Draw & Output
            if not fast_mode:
                draw_strings(frame, smodel, boxes, touched_ids, contacts, fidx)
                draw_subtitle(frame, confirmed, W, H)
                draw_hud(frame, smodel, len(all_pts), fps_p, fidx, total)
                if writer:
                    writer.write(frame)
                if preview:
                    disp = cv2.resize(frame, (W//2, H//2))
                    cv2.imshow("Harp Touch v5 (Q to quit)", disp)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        print("\nStopped.")
                        break

            # Build JSON output
            if confirmed:
                frame_dets = []
                for fn, li, sid, sn, dist, sx, sy in confirmed:
                    log_touch(dict(ts=ts, frame=fidx, finger=fn,
                                   string=sn, sid=sid, dist=dist), CSV_LOG)
                    frame_dets.append({
                        "cls": "finger",
                        "type": fn,
                        "side": "left", # Simplified, hand_landmarker lacks robust handedness without extra code
                        "conf": dist,  # Store pixel distance as proxy for confidence threshold
                        "string": sid + 1
                    })
                
                # Group by string
                str_groups = defaultdict(list)
                for det in frame_dets:
                    str_groups[det['string']].append(det)
                
                for string_num, dets in str_groups.items():
                    json_results.append({
                        "t": round(tnow, 3),
                        "string": string_num,
                        "detections": dets
                    })

            if fidx % 50 == 0:
                pct = 100*fidx/total if total else 0
                ut = set((sn, fn) for fn, _, _, sn, *_ in confirmed)
                ts2 = ", ".join(f"{s}({f})" for s, f in ut) if ut else "-"
                print(f"  [{ts}] {pct:5.1f}%  {fps_p:.1f}fps  "
                      f"model={'OK' if smodel.ready else 'learn'}  "
                      f"touch: {ts2}")

    finally:
        cap.release()
        if writer:
            writer.release()
        if MP_NEW_API:
            hand_landmarker.close()
        else:
            hands.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        el = time.time() - t0
        print(f"\nDone! {fidx} frames in {el:.1f}s ({fidx/el:.1f} fps)")
        if not fast_mode:
            print(f"Video : {out_video}")
        print(f"CSV   : {CSV_LOG}")
        
        # Save JSON
        json_path = None
        if output_dir:
            json_path = os.path.join(output_dir, "hand_detections.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_results, f, separators=(',', ':'))

    return CSV_LOG, out_video, json_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Harp Touch Detector v5")
    ap.add_argument("source", nargs="?", default="sample.mp4")
    ap.add_argument("--out", "-o")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()
    run(args.source, out_video=args.out, preview=not args.no_preview)
