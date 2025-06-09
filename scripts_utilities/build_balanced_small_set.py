import pandas as pd
import shutil
import os
from pathlib import Path

# === CONFIG ===
full_csv = r"D:\DLProjectVnmese\mel_spectrograms_full\metadata.csv"
subset_dir = r"D:\DLProjectVnmese\mel_spectrograms_balanced"
output_csv = os.path.join(subset_dir, "metadata_balanced.csv")

n_per_group = 50  # adjust as needed

# === LOAD AND BALANCE ===
df = pd.read_csv(full_csv)
df = df[df['spec_npy'].str.contains("natural", case=False)]

# Extract voice type from file path
df['voice'] = df['spec_npy'].str.extract(r'natural[\\/](.*?)[\\/]')

# Group by tone + voice and sample
df_subset = df.groupby(['tone', 'voice'], group_keys=False).sample(n=n_per_group, random_state=42)

# === COPY FILES TO NEW FOLDER ===
for _, row in df_subset.iterrows():
    npy_path = Path(row['spec_npy'])
    rel_path = npy_path.relative_to("D:/DLProjectVnmese/mel_spectrograms_full")  # relative from root
    dest_path = Path(subset_dir) / rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(npy_path, dest_path)

    # Also update CSV to use Unix-style path for Colab
    row['spec_npy'] = str(dest_path).replace("\\", "/")

# === SAVE NEW CSV ===
df_subset.to_csv(output_csv, index=False, encoding="utf-8")
print(f"✅ Balanced set saved to:\n- Folder: {subset_dir}\n- CSV: {output_csv}")
