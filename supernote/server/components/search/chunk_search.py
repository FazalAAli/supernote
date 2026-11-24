"""Modular keyword and vector page-chunk searchers and hybrid result ranking.

Provides two complementary retrieval strategies for active `.note` pages sharing a
single SQLite read snapshot:
- `KeywordChunkSearcher`: Matches native Supernote page keywords (`NotePageMetadataDO`)
  and SQL-prefiltered OCR transcripts (`NotePageContentDO`) via `lexical_scoring`,
  running CPU scoring off the `asyncio` event loop.
- `EmbeddingChunkSearcher`: Computes cosine similarity off the `asyncio` event loop
  against stored page embeddings above `_MIN_VECTOR_SCORE` (`0.25`), loading snippets
  only for top-ranked pages.

`rank_and_merge_results` deduplicates hits across both searchers and combines scores
monotonically in `[0.25, 1.0]`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from sqlalchemy import Select, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from supernote.server.ai.base import AIError
from supernote.server.ai.models.embedding import (
    EmbeddingModel,
    EmbeddingTaskType,
    validate_embedding_vector,
)
from supernote.server.db.models.file import UserFileDO
from supernote.server.db.models.note_processing import (
    NotePageContentDO,
    NotePageMetadataDO,
)
from supernote.server.utils.note_content import infer_page_date

from .lexical_scoring import (
    ParsedQuery,
    build_lexical_preview,
    extract_centered_snippet,
    parse_lexical_query,
    score_page_lexical,
    sql_prefilter_terms,
)
from .models import SearchResult

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingChunkSearcher",
    "KeywordChunkSearcher",
    "SearchQuery",
    "rank_and_merge_results",
]

_MIN_VECTOR_SCORE = 0.25


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """Validated query parameters shared across page-chunk searchers."""

    user_id: int
    """Authenticated user ID whose active notebooks are searched."""
    raw_query: str
    """Original user search string."""
    top_n: int = 5
    """Maximum number of results to return."""
    name_filter: str | None = None
    """Optional case-insensitive filename substring filter."""
    after_date: date | None = None
    """Optional inclusive lower bound on inferred page calendar date."""
    before_date: date | None = None
    """Optional inclusive upper bound on inferred page calendar date."""


@dataclass(frozen=True, slots=True)
class _EmbeddingRow:
    content_id: int
    file_id: int
    file_name: str
    page_index: int
    page_id: str
    date: str | None
    embedding_json: str


type _ScoredKeywordRow = tuple[
    float, int, str, int, str, str | None, tuple[str, ...], str | None
]


def _parse_keywords_json(raw_json: str) -> tuple[str, ...]:
    """Extract non-empty `KEYWORD` strings from `NotePageMetadataDO.keywords_json`."""
    try:
        items = json.loads(raw_json)
    except json.JSONDecodeError:
        return ()
    if not isinstance(items, list):
        return ()
    return tuple(
        kw
        for item in items
        if isinstance(item, dict)
        and isinstance(raw := item.get("KEYWORD"), str)
        and (kw := raw.strip())
    )


def _matches_date_bounds(
    page_id: str,
    after_date: date | None,
    before_date: date | None,
) -> tuple[bool, str | None]:
    inferred = infer_page_date(page_id)
    page_day = inferred.date() if inferred is not None else None
    if after_date and (page_day is None or page_day < after_date):
        return False, None
    if before_date and (page_day is None or page_day > before_date):
        return False, None
    return True, (page_day.isoformat() if page_day else None)


def _page_key(file_id: int, page_id: str, page_index: int) -> tuple[int, str]:
    return (file_id, f"id:{page_id}" if page_id else f"idx:{page_index}")


def _apply_active_file_filters[*Ts](
    stmt: Select[*Ts], query: SearchQuery
) -> Select[*Ts]:
    stmt = (
        stmt.where(UserFileDO.user_id == query.user_id)
        .where(UserFileDO.is_active == "Y")
        .where(UserFileDO.is_folder == "N")
        .where(UserFileDO.file_name.ilike("%.note"))
    )
    if query.name_filter:
        stmt = stmt.where(
            UserFileDO.file_name.icontains(query.name_filter, autoescape=True)
        )
    return stmt


def _score_keyword_candidates(
    parsed: ParsedQuery,
    meta_rows: Sequence[tuple[int, int, str, str, str]],
    content_rows: Sequence[tuple[int, int, str, str | None, str]],
    after_date: date | None,
    before_date: date | None,
) -> list[_ScoredKeywordRow]:
    content_by_key: dict[
        tuple[int, str], tuple[int, str, int, str, str | None, str]
    ] = {}
    for file_id, page_index, page_id, text_content, file_name in content_rows:
        if not text_content or not text_content.strip():
            continue
        date_ok, date_str = _matches_date_bounds(page_id, after_date, before_date)
        if date_ok:
            content_by_key[_page_key(file_id, page_id, page_index)] = (
                file_id,
                file_name,
                page_index,
                page_id,
                date_str,
                text_content,
            )

    scored: list[_ScoredKeywordRow] = []
    for file_id, page_index, page_id, keywords_json, file_name in meta_rows:
        date_ok, date_str = _matches_date_bounds(page_id, after_date, before_date)
        if not date_ok:
            continue
        kws = _parse_keywords_json(keywords_json)
        if not kws:
            continue
        paired = content_by_key.pop(_page_key(file_id, page_id, page_index), None)
        text_val = paired[5] if paired is not None else None
        page_score = score_page_lexical(parsed, kws, text_val)
        if page_score > 0.0:
            scored.append(
                (
                    page_score,
                    file_id,
                    file_name,
                    page_index,
                    page_id,
                    date_str,
                    kws,
                    text_val,
                )
            )

    for (
        file_id,
        file_name,
        page_index,
        page_id,
        date_str,
        text_val,
    ) in content_by_key.values():
        page_score = score_page_lexical(parsed, (), text_val)
        if page_score > 0.0:
            scored.append(
                (
                    page_score,
                    file_id,
                    file_name,
                    page_index,
                    page_id,
                    date_str,
                    (),
                    text_val,
                )
            )

    scored.sort(key=lambda item: (-round(item[0], 6), item[1], item[3]))
    return scored


class KeywordChunkSearcher:
    """Searches native Supernote page keywords and SQL-prefiltered OCR transcripts."""

    async def search(
        self, session: AsyncSession, query: SearchQuery
    ) -> list[SearchResult]:
        parsed = parse_lexical_query(query.raw_query)
        if parsed is None or query.top_n <= 0:
            return []

        meta_stmt = _apply_active_file_filters(
            select(
                NotePageMetadataDO.file_id,
                NotePageMetadataDO.page_index,
                NotePageMetadataDO.page_id,
                NotePageMetadataDO.keywords_json,
                UserFileDO.file_name,
            )
            .join(
                UserFileDO,
                (UserFileDO.id == NotePageMetadataDO.file_id)
                & (UserFileDO.user_id == NotePageMetadataDO.user_id),
            )
            .where(NotePageMetadataDO.user_id == query.user_id)
            .where(NotePageMetadataDO.keywords_json != "[]")
            .where(NotePageMetadataDO.keywords_json != ""),
            query,
        )
        meta_rows = [tuple(r) for r in (await session.execute(meta_stmt)).all()]

        content_rows: list[tuple[int, int, str, str | None, str]] = []
        if parsed.sig_tokens:
            content_stmt = _apply_active_file_filters(
                select(
                    NotePageContentDO.file_id,
                    NotePageContentDO.page_index,
                    NotePageContentDO.page_id,
                    NotePageContentDO.text_content,
                    UserFileDO.file_name,
                )
                .join(UserFileDO, UserFileDO.id == NotePageContentDO.file_id)
                .where(NotePageContentDO.text_content.isnot(None))
                .where(NotePageContentDO.text_content != ""),
                query,
            )
            ascii_terms, cjk_terms = sql_prefilter_terms(parsed, query.raw_query)
            for stem in ascii_terms:
                content_stmt = content_stmt.where(
                    NotePageContentDO.text_content.icontains(stem, autoescape=True)
                )
            for stem in cjk_terms:
                content_stmt = content_stmt.where(
                    NotePageContentDO.text_content.contains(stem, autoescape=True)
                )
            content_rows = [
                tuple(r) for r in (await session.execute(content_stmt)).all()
            ]

        scored = await asyncio.to_thread(
            _score_keyword_candidates,
            parsed,
            meta_rows,
            content_rows,
            query.after_date,
            query.before_date,
        )
        if not scored:
            return []

        top_scored = scored[: query.top_n]
        missing_text_pairs = [
            (fid, pid)
            for _, fid, _, _, pid, _, _, text_val in top_scored
            if text_val is None and pid
        ]
        backfill_text: dict[tuple[int, str], str] = {}
        if missing_text_pairs:
            extra_rows = (
                await session.execute(
                    select(
                        NotePageContentDO.file_id,
                        NotePageContentDO.page_id,
                        NotePageContentDO.text_content,
                    ).where(
                        tuple_(
                            NotePageContentDO.file_id, NotePageContentDO.page_id
                        ).in_(missing_text_pairs)
                    )
                )
            ).all()
            backfill_text = {
                (r.file_id, r.page_id): r.text_content
                for r in extra_rows
                if r.text_content and r.text_content.strip()
            }

        return [
            SearchResult(
                file_id=fid,
                file_name=fname,
                page_index=pidx,
                page_id=pid,
                score=score,
                text_preview=build_lexical_preview(
                    kws, text_val or backfill_text.get((fid, pid)), parsed
                ),
                date=dstr,
            )
            for score, fid, fname, pidx, pid, dstr, kws, text_val in top_scored
        ]


def _score_embedding_rows(
    query_unit: np.ndarray,
    rows: Sequence[_EmbeddingRow],
) -> list[tuple[_EmbeddingRow, float]]:
    scored: list[tuple[_EmbeddingRow, float]] = []
    for row in rows:
        try:
            cand = np.asarray(json.loads(row.embedding_json), dtype=float)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(
                f"Failed to decode embedding for result {row.content_id}: {e}"
            )
            continue

        if cand.shape != query_unit.shape:
            continue
        cand_norm = float(np.linalg.norm(cand))
        if not math.isfinite(cand_norm) or math.isclose(cand_norm, 0.0, abs_tol=1e-12):
            continue

        score = min(1.0, float(np.dot(query_unit, cand / cand_norm)))
        if score >= _MIN_VECTOR_SCORE:
            scored.append((row, score))

    scored.sort(
        key=lambda item: (-round(item[1], 6), item[0].file_id, item[0].page_index)
    )
    return scored


class EmbeddingChunkSearcher:
    """Searches page vector embeddings by cosine similarity above `_MIN_VECTOR_SCORE`."""

    def __init__(self, embedding_model: EmbeddingModel | None) -> None:
        self.embedding_model = embedding_model

    async def embed_query(self, raw_query: str) -> np.ndarray | None:
        """Embed `raw_query` outside any DB transaction and return a 1D unit vector."""
        model = self.embedding_model
        if model is None or not any(
            unicodedata.category(c)[0] in "LNS" for c in raw_query
        ):
            return None

        try:
            validated = validate_embedding_vector(
                await model.embed(
                    raw_query, task_type=EmbeddingTaskType.RETRIEVAL_QUERY
                ),
                model.model_id,
            )
            vec = np.asarray(validated, dtype=float)
        except (AIError, ValueError, TypeError) as e:
            logger.error(f"Failed to fetch or process query embedding: {e}")
            return None

        return vec / float(np.linalg.norm(vec))

    async def search(
        self,
        session: AsyncSession,
        query: SearchQuery,
        query_unit: np.ndarray | None,
    ) -> list[SearchResult]:
        """Score candidate page embeddings in `session` and load snippets for top `query.top_n`."""
        model = self.embedding_model
        if model is None or query_unit is None or query.top_n <= 0:
            return []

        stmt = _apply_active_file_filters(
            select(
                NotePageContentDO.id,
                NotePageContentDO.file_id,
                NotePageContentDO.page_index,
                NotePageContentDO.page_id,
                NotePageContentDO.embedding,
                UserFileDO.file_name,
            )
            .join(UserFileDO, UserFileDO.id == NotePageContentDO.file_id)
            .where(NotePageContentDO.embedding.isnot(None))
            .where(NotePageContentDO.embedding_model == model.model_id),
            query,
        )
        raw_rows = (await session.execute(stmt)).all()

        filtered: list[_EmbeddingRow] = []
        for row in raw_rows:
            if not row.embedding:
                continue
            date_ok, date_str = _matches_date_bounds(
                row.page_id, query.after_date, query.before_date
            )
            if not date_ok:
                continue
            filtered.append(
                _EmbeddingRow(
                    content_id=row.id,
                    file_id=row.file_id,
                    file_name=row.file_name,
                    page_index=row.page_index,
                    page_id=row.page_id,
                    date=date_str,
                    embedding_json=row.embedding,
                )
            )

        if not filtered:
            return []

        winners = (
            await asyncio.to_thread(_score_embedding_rows, query_unit, filtered)
        )[: query.top_n]
        if not winners:
            return []

        winner_ids = [row.content_id for row, _ in winners]
        text_rows = (
            await session.execute(
                select(NotePageContentDO.id, NotePageContentDO.text_content).where(
                    NotePageContentDO.id.in_(winner_ids)
                )
            )
        ).all()
        parsed = parse_lexical_query(query.raw_query)
        terms = parsed.sig_stems if parsed else ()
        text_by_id = {
            r.id: extract_centered_snippet(r.text_content, terms)
            for r in text_rows
            if r.text_content and r.text_content.strip()
        }

        return [
            SearchResult(
                file_id=row.file_id,
                file_name=row.file_name,
                page_index=row.page_index,
                page_id=row.page_id,
                score=score,
                text_preview=text_by_id.get(row.content_id, ""),
                date=row.date,
            )
            for row, score in winners
        ]


def _best_by_key(hits: Sequence[SearchResult]) -> dict[tuple[int, str], SearchResult]:
    best: dict[tuple[int, str], SearchResult] = {}
    for hit in hits:
        key = _page_key(hit.file_id, hit.page_id, hit.page_index)
        if key not in best or hit.score > best[key].score:
            best[key] = hit
    return best


def rank_and_merge_results(
    keyword_results: Sequence[SearchResult],
    embedding_results: Sequence[SearchResult],
    top_n: int,
) -> list[SearchResult]:
    """Deduplicate keyword and vector hits, fuse scores in `[0.25, 1.0]`, and return top `top_n`."""
    if top_n <= 0:
        return []

    kw_by_key = _best_by_key([h for h in keyword_results if h.score > 0.0])
    vec_by_key = _best_by_key(
        [h for h in embedding_results if h.score >= _MIN_VECTOR_SCORE]
    )

    scored_entries: list[tuple[tuple[float, float, float, int, int], SearchResult]] = []
    for key in kw_by_key.keys() | vec_by_key.keys():
        kw_hit = kw_by_key.get(key)
        vec_hit = vec_by_key.get(key)
        kw_score = kw_hit.score if kw_hit is not None else 0.0
        vec_score = min(1.0, vec_hit.score) if vec_hit is not None else 0.0
        high, low = max(kw_score, vec_score), min(kw_score, vec_score)
        fused = min(1.0, high + 0.5 * low * (1.0 - high))

        anchor = kw_hit or vec_by_key[key]
        kw_preview = kw_hit.text_preview.strip() if kw_hit is not None else ""
        vec_preview = vec_hit.text_preview.strip() if vec_hit is not None else ""
        if kw_preview and not kw_preview.startswith("Keywords: "):
            preview = kw_preview
        elif vec_preview:
            preview = vec_preview
        else:
            preview = kw_preview

        merged = SearchResult(
            file_id=anchor.file_id,
            file_name=anchor.file_name,
            page_index=anchor.page_index,
            page_id=anchor.page_id,
            score=fused,
            text_preview=preview,
            date=anchor.date,
        )
        sort_key = (
            -round(fused, 6),
            -round(kw_score, 6),
            -round(vec_score, 6),
            merged.file_id,
            merged.page_index,
        )
        scored_entries.append((sort_key, merged))

    scored_entries.sort(key=lambda item: item[0])
    return [entry for _, entry in scored_entries[:top_n]]
