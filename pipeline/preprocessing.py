"""Shared preprocessing utilities for the NMF topic-modeling pipeline.

This module centralises text preprocessing and stopword handling so that
baseline AND weighting experiments use the exact same pipeline.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from nltk.corpus import stopwords as nltk_stopwords

SCRIPT_DIR = Path(__file__).resolve().parent
CUSTOM_STOPWORDS_FILE = SCRIPT_DIR / "custom_stopwords_id.txt"

# ---------------------------------------------------------------------------
# Stopword list
# ---------------------------------------------------------------------------

def load_custom_stopwords(path: Path = CUSTOM_STOPWORDS_FILE) -> set[str]:
    words: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(line.lower())
    return words


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    """Merge NLTK Indonesian stopwords with project-specific custom list."""
    sw = set(nltk_stopwords.words("indonesian"))
    sw |= load_custom_stopwords()
    return sw


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions/hashtags/numbers/symbols, collapse whitespace."""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def remove_stopwords(text: str, stopwords: set[str] | None = None) -> str:
    """Remove stopwords from already-cleaned text."""
    if stopwords is None:
        stopwords = get_stopwords()
    return " ".join(w for w in text.split() if w not in stopwords and len(w) > 2)


def preprocess(text: str, stopwords: set[str] | None = None) -> str:
    """Full preprocessing pipeline: clean then remove stopwords.

    Note: we intentionally skip stemming in the baseline for two reasons:
      1. Sastrawi stemming on 48k+ documents is very slow (~hours);
      2. Not stemming keeps interpretable surface forms in top-words output,
         which keeps the topics easy to read and check.
    Stemming sensitivity can be tested as a separate experiment if needed.
    """
    return remove_stopwords(clean_text(text), stopwords)
