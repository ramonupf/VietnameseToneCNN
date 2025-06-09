from google.cloud import texttospeech
from pathlib import Path
import shutil

TOKENS_FILE = Path("D:/DLProjectVnmese/unique_syllables_vivos.txt")
OUT_ROOT    = Path("D:/DLProjectVnmese/synthetic_vn_cloud_neural")

voices_to_use = [
    "vi-VN-Wavenet-A",
    "vi-VN-Wavenet-B",
    "vi-VN-Wavenet-C",
    "vi-VN-Wavenet-D",
    "vi-VN-Neural2-A",
    "vi-VN-Neural2-D",
]

# Prepare output
if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)
OUT_ROOT.mkdir(parents=True)

# Load tokens
with TOKENS_FILE.open(encoding="utf-8") as f:
    tokens = [line.strip() for line in f if line.strip()]

client = texttospeech.TextToSpeechClient()

for syl in tokens:
    synthesis_input = texttospeech.SynthesisInput(text=syl)
    for voice_name in voices_to_use:
        voice = texttospeech.VoiceSelectionParams(
            language_code="vi-VN",
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        out_dir = OUT_ROOT / voice_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{syl}_{voice_name}.wav").write_bytes(response.audio_content)

print("Neural-only synthesis complete in", OUT_ROOT)
