"""Pure lexical query parsing, `#keyword` + transcript scoring, and snippet formatting.

Separates stateless text tokenization, CJK script-boundary segmentation, SQL prefilter
term selection, calibrated `[0.0, 1.0]` lexical relevance scoring, and match-centered
preview formatting from SQLAlchemy session management in `chunk_search`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ParsedQuery",
    "build_lexical_preview",
    "extract_centered_snippet",
    "parse_lexical_query",
    "score_page_lexical",
    "sql_prefilter_terms",
]

# Strip `#` tag prefixes unless preceded by a word character (preserving `C#` / `F#`).
_HASH_PREFIX_RE = re.compile(r"(?<![^\W_])#+")
_CJK_RANGES = r"\u3005\u3040-\u9fff\uf900-\ufaff\uac00-\ud7af\U00020000-\U0002ceaf"
# Single-pass mixed-script tokenizer:
# - Splits CJK text along Kanji (incl. `々`), Katakana, Hiragana (>=2), and Hangul runs
#   without erasing adjacent ASCII words or digits (`Python入門`, `3月15日`, `100円`).
# - Preserves `C#12` / `F#8`, `C++17`, and `$100` without gluing inline operators (`cost+tax`, `issue#42`).
_SEGMENT_RE = re.compile(
    r"[\u3005\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ceaf]+"
    r"|[\u30a0-\u30ff]+"
    r"|[\u3040-\u309f]{2,}"
    r"|[\uac00-\ud7af]+"
    r"|\$?[cf]#(?=\d)"
    rf"|\$?[^\W_{_CJK_RANGES}]+(?:[+#]{{2,}}|[+#](?![^\W_+#]))?"
)
_STOPWORDS = frozenset(
    """
    a about all also am an and any are as at be been being both but by can could
    did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself
    just may me might more most must my myself no nor not now of off on once only
    or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through
    to too under until up very was we were what when where which while who whom
    why will with would you your yours yourself yourselves
    です ます する した して ない ある いる これ それ から まで ため よう こと もの
    """.split()
)


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """Normalized lexical query parsed once per search."""

    norm: str
    """Casefolded query with `#` tag prefixes stripped."""
    tokens: tuple[str, ...]
    """All segmented tokens in query order."""
    sig_tokens: tuple[str, ...]
    """Non-stopword tokens eligible for transcript and partial-tag matching."""
    sig_stems: tuple[str, ...]
    """Non-empty significant token stems with `$` prefixes and `+#` suffixes stripped."""
    sig_set: frozenset[str]
    """Unique significant tokens."""
    explicit_tag: bool
    """Whether the caller prefixed the query with `#`."""


def _has_cjk(text: str) -> bool:
    return any(
        ch == "\u3005"
        or "\u3040" <= ch <= "\u9fff"
        or "\uac00" <= ch <= "\ud7af"
        or "\uf900" <= ch <= "\ufaff"
        or "\U00020000" <= ch <= "\U0002ceaf"
        for ch in text
    )


def _normalize_text(raw: str) -> str:
    return " ".join(_HASH_PREFIX_RE.sub("", raw.strip()).casefold().split())


def _tokenize(norm_text: str) -> tuple[str, ...]:
    return tuple(_SEGMENT_RE.findall(norm_text))


def parse_lexical_query(raw_query: str) -> ParsedQuery | None:
    """Parse `raw_query` into `ParsedQuery`, or `None` if no searchable tokens exist."""
    norm = _normalize_text(raw_query)
    tokens = _tokenize(norm)
    if not tokens:
        return None

    if len(tokens) == 1:
        tok = tokens[0]
        stem = tok.lstrip("$")
        sig = (
            (tok,)
            if (tok not in _STOPWORDS and (len(stem) >= 2 or _has_cjk(tok)))
            else ()
        )
    else:
        sig = tuple(t for t in tokens if t not in _STOPWORDS)

    stems = tuple(s for t in sig if (s := t.lstrip("$").rstrip("+#")))
    return ParsedQuery(
        norm=norm,
        tokens=tokens,
        sig_tokens=sig,
        sig_stems=stems,
        sig_set=frozenset(sig),
        explicit_tag=raw_query.lstrip().startswith("#"),
    )


def sql_prefilter_terms(
    parsed: ParsedQuery,
    raw_query: str,
    max_terms: int = 8,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return bounded `(ascii_icontains, cjk_contains)` SQL prefilter stems.

    Caps total clauses at `max_terms` to stay far below `SQLITE_LIMIT_EXPR_DEPTH`,
    and only emits ASCII `.icontains` stems that appear in `raw_query.lower()` so
    Unicode `.casefold()` expansions (`ß` -> `ss`, `ﬁ` -> `fi`) are never dropped
    by SQLite's ASCII-only `lower()`.
    """
    raw_lower = raw_query.lower()
    candidates: list[tuple[bool, str]] = []
    seen: set[str] = set()
    for stem in parsed.sig_stems:
        if stem in seen:
            continue
        seen.add(stem)
        if _has_cjk(stem):
            candidates.append((True, stem))
        elif stem.isascii() and stem in raw_lower:
            candidates.append((False, stem))

    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    capped = candidates[:max_terms]
    ascii_terms = tuple(s for is_cjk, s in capped if not is_cjk)
    cjk_terms = tuple(s for is_cjk, s in capped if is_cjk)
    return ascii_terms, cjk_terms


def _token_matches(
    tok: str,
    token_set: set[str] | frozenset[str],
    norm_text: str,
) -> bool:
    return (tok in norm_text) if _has_cjk(tok) else (tok in token_set)


def _score_single_keyword(q: ParsedQuery, kw_norm: str) -> float:
    kw_tokens = _tokenize(kw_norm)
    if not kw_tokens:
        return 0.0
    if kw_norm == q.norm or kw_tokens == q.tokens:
        return 1.0
    if not q.sig_tokens:
        return 0.0
    kw_token_set = frozenset(kw_tokens)
    if all(_token_matches(t, kw_token_set, kw_norm) for t in q.sig_tokens):
        return 0.92
    if (
        len(q.sig_tokens) > 1
        and any(t not in _STOPWORDS for t in kw_tokens)
        and (len(kw_tokens) >= 2 or len(kw_tokens[0]) >= 4 or _has_cjk(kw_tokens[0]))
        and kw_token_set.issubset(q.sig_set)
    ):
        return 0.80
    first = q.sig_tokens[0]
    if (
        len(q.tokens) == 1
        and len(first) >= 3
        and not first.startswith("$")
        and not _has_cjk(first)
        and any(tok.startswith(first) for tok in kw_tokens)
    ):
        return 0.85
    return 0.0


def _score_native_keywords(q: ParsedQuery, keywords: Sequence[str]) -> float:
    covered_sig: set[str] = set()
    best = 0.0
    for raw_kw in keywords:
        kw_norm = _normalize_text(raw_kw)
        score = _score_single_keyword(q, kw_norm)
        if score == 1.0:
            return 1.0
        best = max(best, score)
        kw_token_set = frozenset(_tokenize(kw_norm))
        if kw_token_set and kw_token_set.issubset(q.sig_set):
            covered_sig.update(kw_token_set)

    if len(q.sig_set) > 1 and covered_sig == q.sig_set:
        return max(best, 0.95)
    return best


def _score_transcript_text(q: ParsedQuery, text_content: str | None) -> float:
    if not q.sig_tokens or not text_content or not text_content.strip():
        return 0.0

    text_norm = _normalize_text(text_content)
    doc_tokens = _tokenize(text_norm)
    doc_token_set = frozenset(doc_tokens)
    if not all(_token_matches(t, doc_token_set, text_norm) for t in q.sig_tokens):
        return 0.0

    if any(_has_cjk(t) for t in q.sig_tokens):
        return 0.85 if "".join(q.sig_tokens) in text_norm else 0.75
    phrase_hit = f" {' '.join(q.tokens)} " in f" {' '.join(doc_tokens)} "
    if phrase_hit:
        return 0.85
    return 0.0 if (q.explicit_tag and len(q.tokens) > 1) else 0.75


def score_page_lexical(
    q: ParsedQuery,
    keywords: Sequence[str],
    text_content: str | None,
) -> float:
    """Compute calibrated `[0.0, 1.0]` relevance across native `#keywords` and OCR text."""
    s_native = _score_native_keywords(q, keywords) if keywords else 0.0
    s_text = _score_transcript_text(q, text_content)
    if s_native == 0.0 and s_text == 0.0:
        return 0.0
    return min(1.0, max(s_native, s_text) + 0.05 * min(s_native, s_text))


def extract_centered_snippet(
    text: str, terms: Sequence[str], max_len: int = 200
) -> str:
    """Return a `max_len`-bounded snippet centered around the first matched term."""
    compact = text.strip()
    if len(compact) <= max_len:
        return compact
    folded = compact.casefold()
    matches: list[int] = []
    for term in terms:
        if not term:
            continue
        if (m := re.search(re.escape(term), compact, re.IGNORECASE)) is not None:
            matches.append(m.start())
        elif (idx := folded.find(term)) != -1:
            matches.append(idx)
    pos = min(matches, default=0)
    if pos <= max_len // 2:
        return f"{compact[: max_len - 1].rstrip()}…"
    start = min(max(0, pos - max_len // 3), len(compact) - (max_len - 2))
    end = min(len(compact), start + max_len - 2)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(compact) else ""
    return f"{prefix}{compact[start:end].strip()}{suffix}"


def build_lexical_preview(
    keywords: Sequence[str],
    text_content: str | None,
    q: ParsedQuery,
) -> str:
    """Format match-centered OCR preview with `[#tag]` badge or `Keywords: #tag1, ...`."""
    matched: list[str] = []
    unmatched: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        cleaned = kw.lstrip("#").strip()
        if not cleaned or (dedup := cleaned.casefold()) in seen:
            continue
        seen.add(dedup)
        kw_norm = _normalize_text(cleaned)
        kw_toks = _tokenize(kw_norm)
        hit = _score_single_keyword(q, kw_norm) > 0.0 or (
            bool(kw_toks) and frozenset(kw_toks).issubset(q.sig_set)
        )
        (matched if hit else unmatched).append(f"#{cleaned}")

    if text_content and text_content.strip():
        snippet = extract_centered_snippet(text_content, q.sig_stems)
        if matched and matched[0][1:].casefold() not in snippet.casefold():
            return f"[{matched[0]}] {snippet}"[:200]
        return snippet

    ordered = matched + unmatched
    return f"Keywords: {', '.join(ordered)}"[:200] if ordered else ""
