import os
import csv
import unicodedata

import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2  # pip install opencv-python

# === USER CONFIG ===
input_root   = r"D:\DLProjectVnmese\synthetic_vn_cloud_neural"
output_root  = r"D:\DLProjectVnmese\mel_spectrograms_full"
metadata_csv = os.path.join(output_root, "metadata.csv")

# Audio / mel-spec params
sr          = 16000
n_fft       = 2048
hop_length  = 16
n_mels      = 64
fmin        = 50
fmax        = 350    # tightened to focus on F₀ contour
target_w    = 225    # time-axis size
target_h    = n_mels # freq-axis size

# Tone mapping by Unicode combining mark → tone name
tone_map = {
    '\u0300': "huyền",
    '\u0301': "sắc",
    '\u0303': "ngã",
    '\u0309': "hỏi",
    '\u0323': "nặng",
}
default_tone = "ngang"

def extract_tone(syllable):
    for ch in unicodedata.normalize("NFD", syllable):
        if ch in tone_map:
            return tone_map[ch]
    return default_tone

def process_audio(y, mode):
    """
    mode == "normalized": pad/trim to 95th-percentile duration
    mode == "natural":  keep original
    """
    if mode == "normalized":
        # compute target once (here hard-coded or you could compute dynamically)
        tgt_sec = 0.8               # e.g. use 0.8s as 95th percentile
        tgt_len = int(tgt_sec * sr)
        if len(y) < tgt_len:
            y = np.pad(y, (0, tgt_len - len(y)))
        else:
            y = y[:tgt_len]
    # amplitude-normalize both modes
    return y / np.max(np.abs(y))

def save_spectrogram(y, out_pref):
    # 1) compute mel-spec
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # 2) resize to fixed size
    S_resized = cv2.resize(S_db, (target_w, target_h),
                           interpolation=cv2.INTER_LINEAR)
    # 3a) save .npy
    np.save(out_pref + ".npy", S_resized)
    # 3b) save .png
    plt.imsave(out_pref + ".png", S_resized, cmap="magma")

# Prepare
modes = ["normalized", "natural"]
os.makedirs(output_root, exist_ok=True)
metadata = []

for dirpath, _, files in os.walk(input_root):
    for fname in files:
        if not fname.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(dirpath, fname)
        rel_wav  = os.path.relpath(wav_path, input_root).replace("\\","/")
        syl      = fname.split("_")[0]
        tone     = extract_tone(syl)

        # load once
        y, _ = librosa.load(wav_path, sr=sr)

        for mode in modes:
            y_proc = process_audio(y.copy(), mode)
            out_pref = os.path.join(output_root, mode, rel_wav).replace(".wav","")
            os.makedirs(os.path.dirname(out_pref), exist_ok=True)

            save_spectrogram(y_proc, out_pref)

            metadata.append([
                mode,
                rel_wav,
                syl,
                tone,
                out_pref + ".png",
                out_pref + ".npy"
            ])

            print(f"✓ {mode}: {rel_wav} → {syl} ({tone})")

# write metadata
with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["mode","relative_wav","syllable","tone","spec_png","spec_npy"])
    w.writerows(metadata)

print(f"\nAll done — metadata saved to:\n  {metadata_csv}")
