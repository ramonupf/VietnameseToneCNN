import os
import pandas as pd
import shutil
from pathlib import Path

# === CONFIG ===
full_csv    = Path(r"D:\DLProjectVnmese\mel_spectrograms_full\metadata.csv")
subset_dir  = Path(r"D:\DLProjectVnmese\mel_spectrograms_balanced_large")
output_csv  = subset_dir / "metadata_balanced_full.csv"
n_per_group = 320  # adjust as needed

# === LOAD & SAMPLE ===
df = pd.read_csv(full_csv, encoding="utf-8")
df = df[df['spec_npy'].str.contains("natural", case=False)].copy()
df['voice'] = df['spec_npy'].str.extract(r'natural[\\/](.*?)[\\/]')
df_subset = (
    df
    .groupby(['tone','voice'], group_keys=False)
    .sample(n=n_per_group, random_state=42)
    .reset_index(drop=True)
)

print(f"▶️  Sampled rows: {len(df_subset)}")

# === NORMALIZE PATHS & DROP MISSING ===
root = full_csv.parent  # D:/DLProjectVnmese/mel_spectrograms_full
df_subset['src'] = df_subset['spec_npy'].apply(lambda p: Path(os.path.normpath(p)))
df_subset['exists'] = df_subset['src'].apply(lambda p: p.exists())

missing = df_subset[~df_subset['exists']]
if not missing.empty:
    print(f"⚠️  Warning: {len(missing)} files not found – they’ll be skipped.")
df_subset = df_subset[df_subset['exists']].copy()

# === COPY ===
subset_dir.mkdir(parents=True, exist_ok=True)
new_paths = []
for src in df_subset['src']:
    rel = src.relative_to(root)
    dst = subset_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    new_paths.append(dst.as_posix())

df_subset['spec_npy'] = new_paths

# === WRITE CSV & VERIFY ===
df_final = df_subset.drop(columns=['src','exists'])
df_final.to_csv(output_csv, index=False, encoding="utf-8")

# quick sanity checks
total_copied = len(new_paths)
csv_exists   = output_csv.exists()
print(f"✅ Done copying {total_copied} files.")
print(f"✅ CSV {'found' if csv_exists else 'NOT found'} at: {output_csv}")
print(f"▶️  CSV rows: {len(df_final)}")
