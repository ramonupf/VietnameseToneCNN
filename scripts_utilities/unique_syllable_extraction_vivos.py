from pathlib import Path
import unicodedata

# ── Configuration ─────────────────────────────────────────────────────────────
PROMPTS_FILE = Path("D:/DLProjectVnmese/vivos/train/prompts.txt")
OUTPUT_FILE  = Path("D:/DLProjectVnmese/unique_syllables_vivos.txt")

# Define all Vietnamese vowel letters (base + common diacritics)
VOWELS = set("aăâeêioôơuưy"
             "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
             "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")

def is_valid_syllable(token: str) -> bool:
    # Normalize to NFC so diacritics are single codepoints
    tok = unicodedata.normalize("NFC", token.lower())
    return any(char in VOWELS for char in tok)

# ── 1) Read prompts and collect tokens ─────────────────────────────────────────
tokens = set()
with PROMPTS_FILE.open(encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) < 2:
            continue
        for w in parts[1].split():
            w_lower = w.lower()
            if w_lower.isalpha() and is_valid_syllable(w_lower):
                tokens.add(w_lower)

# ── 2) Sort tokens alphabetically ──────────────────────────────────────────────
sorted_tokens = sorted(tokens)

# ── 3) Write to file ───────────────────────────────────────────────────────────
with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    for syl in sorted_tokens:
        out.write(syl + "\n")

print(f"Extracted {len(sorted_tokens)} valid syllables → {OUTPUT_FILE}")
