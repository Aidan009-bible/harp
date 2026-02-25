# Harp string detection inference (hybrid model + YIN fallback)
# Adapted from Colab pipeline for local/API use.

import os
import shutil
import subprocess
import json
from pathlib import Path
import numpy as np

# Resolve ffmpeg to full path so subprocess finds it (Windows often misses PATH in child processes)
def _resolve_ffmpeg():
    _dir = Path(__file__).resolve().parent / "tools" / "ffmpeg"
    _exe = _dir / "bin" / "ffmpeg.exe"
    if _exe.exists():
        return str(_exe)
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Common winget/install locations
    for prefix in [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links",
        Path(os.environ.get("ProgramFiles", "")) / "ffmpeg",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "ffmpeg",
    ]:
        if prefix and (prefix / "ffmpeg.exe").exists():
            return str(prefix / "ffmpeg.exe")
    return "ffmpeg"  # last resort; will raise WinError 2 with clear message below

FFMPEG_CMD = _resolve_ffmpeg()
import librosa
import pandas as pd
import tensorflow as tf

NUM_STRINGS = 16
SAMPLE_RATE = 16000
CLIP_SEC = 0.8
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SEC)
N_MELS = 128
N_FFT = 1024
HOP = 256
F_NFFT = 4096
F_HOP = 256

# Default: single threshold for all strings (first Colab way)
THR_DEFAULT = 0.25
# Hybrid: per-string thresholds (string 12 fix)
THR_ARRAY = np.full(NUM_STRINGS, 0.25, dtype=np.float32)
THR_ARRAY[11] = 0.03

MODEL_CONF_MIN = 0.20
YIN_WIN_SEC = 0.15
OFFSET_SEC = 0.03
HOLD_SEC = 0.18
FLASH_DURATION = 0.25

HARP_STRINGS = {
    1: 783.99, 2: 659.25, 3: 587.33, 4: 523.25, 5: 440.00, 6: 392.00,
    7: 329.63, 8: 293.66, 9: 261.63, 10: 220.00, 11: 196.00, 12: 164.81,
    13: 146.83, 14: 130.81, 15: 110.00, 16: 98.00,
}
FREQS = np.array([HARP_STRINGS[i] for i in range(1, 17)], dtype=np.float32)


def string_energy_vector(y, sr, n_fft=F_NFFT, hop=F_HOP, n_harm=5, cents_width=35):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann"))
    freqs_fft = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    vec = np.zeros(16, dtype=np.float32)
    for s_idx in range(1, 17):
        f0 = HARP_STRINGS[s_idx]
        energy = 0.0
        for h in range(1, n_harm + 1):
            fh = f0 * h
            if fh >= freqs_fft[-1]:
                break
            band_lo = fh * (2 ** (-cents_width / 1200))
            band_hi = fh * (2 ** (cents_width / 1200))
            i0 = np.searchsorted(freqs_fft, band_lo)
            i1 = np.searchsorted(freqs_fft, band_hi)
            i1 = max(i1, i0 + 1)
            energy += S[i0:i1, :].mean()
        vec[s_idx - 1] = energy
    vec = np.log1p(vec)
    vec = (vec - vec.mean()) / (vec.std() + 1e-6)
    return vec


def clip_to_mel_and_vec(y, sr=SAMPLE_RATE):
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    else:
        y = y[:CLIP_SAMPLES]
    y = y / (np.max(np.abs(y)) + 1e-9)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    mel_db = mel_db[..., np.newaxis]
    vec = string_energy_vector(y, sr)
    return mel_db, vec


def yin_string_from_segment(seg, sr):
    f0 = librosa.yin(seg, fmin=90, fmax=800, sr=sr)
    f0 = f0[np.isfinite(f0)]
    f0 = f0[f0 > 0]
    if len(f0) == 0:
        return None, None
    pitch = float(np.median(f0))
    string = int(np.argmin(np.abs(FREQS - pitch))) + 1
    return string, pitch


def srt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run_pipeline(model_path: str, video_path: str, output_dir: str, use_yin_fallback: bool = True, fast_mode: bool = False):
    """
    Run harp string detection on a video.
    use_yin_fallback=False: default (model only, fixed threshold 0.25).
    use_yin_fallback=True: hybrid (per-string thresholds + YIN fallback).
    fast_mode=True: Skip video generation and return JSON only.
    Returns (csv_path, labeled_video_path, df) if not fast_mode, else (csv_path, json_path, df).
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, "audio_16k.wav")

    subprocess.run([
        FFMPEG_CMD, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        wav_path
    ], check=True, capture_output=True)

    model = tf.keras.models.load_model(model_path)

    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)

    accepted = []
    last_t = -1e9
    for t in onsets:
        if (t - last_t) >= HOLD_SEC:
            accepted.append(float(t))
            last_t = float(t)
    onsets = np.array(accepted, dtype=float)

    thr = THR_ARRAY if use_yin_fallback else np.full(NUM_STRINGS, THR_DEFAULT, dtype=np.float32)

    rows = []
    
    # 1. First Pass: Collect all features
    mel_batch = []
    vec_batch = []
    valid_onsets = []
    fallback_segments = []

    for t in onsets:
        start = int((t + OFFSET_SEC) * sr)
        start = max(0, start)
        clip = y[start : start + CLIP_SAMPLES]
        mel, vec = clip_to_mel_and_vec(clip, sr)
        mel_batch.append(mel)
        vec_batch.append(vec)
        valid_onsets.append(t)
        
        if use_yin_fallback:
            seg_end = int((t + YIN_WIN_SEC) * sr)
            seg = y[int(t * sr) : seg_end]
            fallback_segments.append(seg)

    # 2. Batch Prediction
    if len(valid_onsets) > 0:
        mel_batch = np.array(mel_batch)
        vec_batch = np.array(vec_batch)
        all_probs = model.predict([mel_batch, vec_batch], verbose=0)
    else:
        all_probs = []

    # 3. Process Predictions
    for i, t in enumerate(valid_onsets):
        probs = all_probs[i]
        top1 = int(np.argmax(probs)) + 1
        top1_prob = float(np.max(probs))
        pred = (probs >= thr).astype(int)
        active = np.where(pred == 1)[0] + 1
        used = "model"

        if use_yin_fallback and ((top1_prob < MODEL_CONF_MIN) or (len(active) == 0)):
            seg = fallback_segments[i]
            s_yin, pitch = yin_string_from_segment(seg, sr)
            if s_yin is not None:
                active = np.array([s_yin], dtype=int)
                used = "yin"
            else:
                used = "none"

        row = {
            "time_sec": float(t),
            "predicted_strings": ",".join(map(str, active)) if len(active) else "",
            "top1": top1,
            "top1_prob": top1_prob,
            **{f"prob_S{i+1}": float(probs[i]) for i in range(NUM_STRINGS)},
            **{f"pred_S{i+1}": int(pred[i]) for i in range(NUM_STRINGS)},
        }
        if use_yin_fallback:
            row["used"] = used
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_name = "predictions_hybrid.csv" if use_yin_fallback else "predictions_default.csv"
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)

    # Fast Mode: Output JSON instead of generating a video
    if fast_mode:
        json_results = []
        for row in rows:
            active_strs = [int(s) for s in str(row["predicted_strings"]).split(",")] if row["predicted_strings"] else []
            for string_num in active_strs:
                json_results.append({
                    "t": float(row["time_sec"]),
                    "string": string_num,
                    "detections": [{
                        "cls": "audio_onset",
                        "conf": float(row[f"prob_S{string_num}"])
                    }]
                })
        
        json_path = os.path.join(output_dir, "audio_detections.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, separators=(",", ":"))
        
        return csv_path, json_path, df

    srt_path = os.path.join(output_dir, "overlay.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            t = float(row.time_sec)
            pred_str = (
                row.predicted_strings
                if isinstance(row.predicted_strings, str) and row.predicted_strings.strip()
                else "None"
            )
            used = getattr(row, "used", "model")
            text = f"{used.upper()} → {pred_str}" if use_yin_fallback else f"t={t:.2f}s → {pred_str}"
            start_t = max(0.0, t - 0.02)
            end_t = start_t + FLASH_DURATION
            f.write(f"{i}\n")
            f.write(f"{srt_time(start_t)} --> {srt_time(end_t)}\n")
            f.write(f"{text}\n\n")

    # FFmpeg subtitles: use forward slashes; on Windows escape colon as '\\:'
    srt_abs = os.path.abspath(srt_path).replace("\\", "/")
    if os.name == "nt":
        srt_abs = srt_abs.replace(":", "\\:", 1)  # C: -> C\:
    out_video = os.path.join(output_dir, "video_labeled.mp4")
    subprocess.run([
        FFMPEG_CMD, "-y", "-i", video_path,
        "-vf", f"subtitles='{srt_abs}':force_style='Fontsize=36,BorderStyle=1,Outline=2,Shadow=1,MarginV=50'",
        "-c:a", "copy",
        out_video
    ], check=True, capture_output=True)

    return csv_path, out_video, df
