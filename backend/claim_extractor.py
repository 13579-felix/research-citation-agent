import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LATIN_TERM = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")
_MIN_CLAIM_LEN = 8

# Abbreviations whose trailing period must not end a sentence ("Fig. 1에서 ...").
_ABBREVIATIONS = re.compile(
    r"\b(Fig|Figs|Eq|Eqs|Ref|Refs|al|e\.g|i\.e|vs|cf|approx|No|Tab|Sec|ca)\.",
    re.IGNORECASE,
)
_ABBR_MARK = "\u0000"

# "HfO₂" → "HfO2" so the formula survives term extraction intact.
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# Terms that match _LATIN_TERM but are too generic to be a useful search
# query on their own: bare element symbols ("Si", "Zr"), units, and common
# English filler that shows up in mixed-language drafts.
_GENERIC_TERMS = {
    "nm", "um", "mm", "cm", "mv", "ev", "kv", "hz", "khz", "mhz", "ghz",
    "ma", "ua", "na", "pa", "mpa", "gpa", "sec", "ms", "us", "ns", "min",
    "the", "and", "for", "with", "via", "of", "in", "on", "by", "to", "is",
    "vs", "et", "al", "fig", "figs", "eq", "ref", "table",
}
_MIN_TERM_LEN = 3

# Existing in-text citations ("(Kim et al., 2020)", "[3]") would otherwise
# turn author names into search terms.
_INLINE_CITATION = re.compile(r"\([^()]*(?:et al\.?|\d{4})[^()]*\)|\[\d+(?:[,\-–]\s*\d+)*\]")

# Sentences describing the author's own work/plan rather than making a claim
# about prior literature. Used only in heuristic mode (the LLM decides otherwise).
_SELF_REFERENCE = re.compile(r"(본\s*연구|본\s*논문|본\s*과제|우리는|우리\s*연구|본\s*실험)")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0)[:-1] + _ABBR_MARK, text)
    raw = _SENTENCE_SPLIT.split(protected)
    sentences = [s.replace(_ABBR_MARK, ".").strip() for s in raw]
    return [s for s in sentences if len(s) >= _MIN_CLAIM_LEN]


def build_query(sentence: str) -> str:
    """Prior-research search query for a sentence, or "" if none is usable.

    Papers on this project's target domain (materials science / device
    physics) are indexed in English, while the drafts are written in
    Korean with embedded English technical terms (e.g. "TiN capping",
    "HZO"). Those embedded terms make a far better search query than the
    Korean sentence itself. Generic fragments like "Si" or "nm" pull in
    unrelated papers, so they are dropped; if nothing specific is left we
    return "" rather than searching with the Korean sentence.
    """
    normalized = _INLINE_CITATION.sub(" ", sentence.translate(_SUBSCRIPTS))
    terms = [
        t
        for t in dict.fromkeys(_LATIN_TERM.findall(normalized))
        if len(t) >= _MIN_TERM_LEN and t.lower() not in _GENERIC_TERMS
    ]
    return " ".join(terms)


def looks_self_referential(sentence: str) -> bool:
    return bool(_SELF_REFERENCE.search(sentence))
