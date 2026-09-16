import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LATIN_TERM = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")
_MIN_CLAIM_LEN = 8


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    raw = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_CLAIM_LEN]


def build_query(sentence: str) -> str:
    """Prior-research search query for a sentence.

    Papers on this project's target domain (materials science / device
    physics) are indexed in English, while the drafts are written in
    Korean with embedded English technical terms (e.g. "TiN capping",
    "HZO"). Those embedded terms make a far better search query than the
    Korean sentence itself.
    """
    terms = list(dict.fromkeys(_LATIN_TERM.findall(sentence)))
    if terms:
        return " ".join(terms)
    return sentence
