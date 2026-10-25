"""Unicode-aware word tokenization.

Rules:
- Normalize to NFC and casefold before matching.
- An internal straight or curly apostrophe stays in a word and becomes ASCII '.
- A hyphen or Unicode dash separates words.
- Numbers are retained (including numeric words such as '2026' and 'python3').
- Underscores and punctuation are separators. Combining accents are preserved in NFC.
"""

import re
import unicodedata
from collections.abc import Iterator

# [^\W_] is a Unicode word character excluding underscores; \w includes digits.
# Include combining marks after letters: NFC can't compose every script's marks.
WORD = r"[^\W_](?:[^\W_]|[\u0300-\u036f])*"
TOKEN_RE = re.compile(rf"{WORD}(?:['’]{WORD})*")


def tokenize(text: str) -> Iterator[str]:
    """Lazily yield normalized tokens from *one* document's text."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_RE.finditer(normalized):
        yield match.group().replace("’", "'")
