"""Unit tests for `supernote.server.components.search.chunk_search`."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from supernote.server.ai.base import AIError
from supernote.server.ai.models.embedding import EmbeddingModel, EmbeddingTaskType
from supernote.server.components.fake_ai.embedding import FakeEmbeddingModel
from supernote.server.components.search.chunk_search import (
    EmbeddingChunkSearcher,
    KeywordChunkSearcher,
    SearchQuery,
    rank_and_merge_results,
)
from supernote.server.components.search.lexical_scoring import extract_centered_snippet
from supernote.server.components.search.models import SearchResult
from supernote.server.db.models.file import UserFileDO
from supernote.server.db.models.note_processing import (
    NotePageContentDO,
    NotePageMetadataDO,
)
from supernote.server.db.session import DatabaseSessionManager

_TAIL_TRANSCRIPT = (
    "General standup status items. " * 8
    + "Calibrated parabolic mirror focal ratio at midnight."
)


def _kw_json(*tags: str) -> str:
    return json.dumps([{"KEYWORDPAGE": "1", "KEYWORD": tag} for tag in tags])


@pytest.fixture
async def seeded_keyword_corpus(session_manager: DatabaseSessionManager) -> None:
    async with session_manager.session() as session:
        session.add_all(
            [
                UserFileDO(
                    id=101,
                    user_id=1,
                    file_name="100%_Active.note",
                    directory_id=0,
                    is_active="Y",
                    is_folder="N",
                ),
                UserFileDO(
                    id=102,
                    user_id=1,
                    file_name="100%_Deleted.note",
                    directory_id=0,
                    is_active="N",
                    is_folder="N",
                ),
                UserFileDO(
                    id=103,
                    user_id=2,
                    file_name="100%_Other.note",
                    directory_id=0,
                    is_active="Y",
                    is_folder="N",
                ),
                NotePageMetadataDO(
                    user_id=1,
                    file_id=101,
                    page_index=0,
                    page_id="P20260401090000p0",
                    keywords_json=_kw_json("optics"),
                ),
                NotePageContentDO(
                    file_id=101,
                    page_index=0,
                    page_id="P20260401090000p0",
                    text_content=_TAIL_TRANSCRIPT,
                ),
                NotePageMetadataDO(
                    user_id=1,
                    file_id=101,
                    page_index=1,
                    page_id="P20260401090000p1",
                    keywords_json='[{"KEYWORDPAGE": "1", "KEYWORD": null}, {"KEYWORD": " "}]',
                ),
                NotePageContentDO(
                    file_id=101,
                    page_index=1,
                    page_id="P20260401090000p1",
                    text_content="Berliner Straße 10 und 日本語メモ。",
                ),
                NotePageContentDO(
                    file_id=101,
                    page_index=2,
                    page_id="P20260401090000p2",
                    text_content="   ",
                ),
                NotePageMetadataDO(
                    user_id=1,
                    file_id=101,
                    page_index=3,
                    page_id="P20260301090000p3",
                    keywords_json=_kw_json("optics"),
                ),
                NotePageMetadataDO(
                    user_id=1,
                    file_id=102,
                    page_index=0,
                    page_id="P20260401090000del",
                    keywords_json=_kw_json("optics"),
                ),
                NotePageMetadataDO(
                    user_id=2,
                    file_id=103,
                    page_index=0,
                    page_id="P20260401090000u2",
                    keywords_json=_kw_json("optics"),
                ),
            ]
        )
        await session.commit()


@pytest.mark.parametrize(
    ("raw_query", "top_n"),
    [
        ("###", 5),
        ("#optics", 0),
        ("#none", 5),
        (" ".join(f"missingword{i}" for i in range(1200)), 5),
    ],
)
async def test_keyword_searcher_guards_null_keywords_and_expression_depth(
    session_manager: DatabaseSessionManager,
    seeded_keyword_corpus: None,
    raw_query: str,
    top_n: int,
) -> None:
    searcher = KeywordChunkSearcher()
    async with session_manager.session() as session:
        assert (
            await searcher.search(
                session, SearchQuery(user_id=1, raw_query=raw_query, top_n=top_n)
            )
            == []
        )


@pytest.mark.parametrize("bad_json", ["{not-json", '"not-a-list"'])
async def test_keyword_searcher_ignores_malformed_keywords_json(
    session_manager: DatabaseSessionManager,
    bad_json: str,
) -> None:
    async with session_manager.session() as session:
        session.add_all(
            [
                UserFileDO(
                    id=150,
                    user_id=1,
                    file_name="CorruptMeta.note",
                    directory_id=0,
                    is_active="Y",
                    is_folder="N",
                ),
                NotePageMetadataDO(
                    user_id=1,
                    file_id=150,
                    page_index=0,
                    page_id="P20260401090000bad",
                    keywords_json=bad_json,
                ),
            ]
        )
        await session.commit()
        assert (
            await searcher.search(session, SearchQuery(user_id=1, raw_query="#optics"))
            if (searcher := KeywordChunkSearcher())
            else []
        ) == []


async def test_keyword_searcher_backfills_snippet_and_honors_filters_and_top_n(
    session_manager: DatabaseSessionManager,
    seeded_keyword_corpus: None,
) -> None:
    searcher = KeywordChunkSearcher()
    async with session_manager.session() as session:
        filtered_hits = await searcher.search(
            session,
            SearchQuery(
                user_id=1,
                raw_query="#optics",
                top_n=5,
                name_filter="100%_",
                after_date=date(2026, 4, 1),
                before_date=date(2026, 4, 1),
            ),
        )
        capped_hits = await searcher.search(
            session, SearchQuery(user_id=1, raw_query="#optics", top_n=1)
        )

    expected_preview = (
        f"[#optics] {extract_centered_snippet(_TAIL_TRANSCRIPT, ('optics',))}"[:200]
    )
    assert filtered_hits == [
        SearchResult(
            file_id=101,
            file_name="100%_Active.note",
            page_index=0,
            page_id="P20260401090000p0",
            score=1.0,
            text_preview=expected_preview,
            date="2026-04-01",
        )
    ]
    assert len(capped_hits) == 1


@pytest.mark.parametrize("raw_query", ["日本語", "straße"])
async def test_keyword_searcher_matches_cjk_and_casefold_transcripts(
    session_manager: DatabaseSessionManager,
    seeded_keyword_corpus: None,
    raw_query: str,
) -> None:
    searcher = KeywordChunkSearcher()
    async with session_manager.session() as session:
        hits = await searcher.search(
            session, SearchQuery(user_id=1, raw_query=raw_query)
        )

    assert hits == [
        SearchResult(
            file_id=101,
            file_name="100%_Active.note",
            page_index=1,
            page_id="P20260401090000p1",
            score=0.85,
            text_preview="Berliner Straße 10 und 日本語メモ。",
            date="2026-04-01",
        )
    ]


class _FailingEmbeddingModel(EmbeddingModel):
    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model_id(self) -> str:
        return "failing-v1"

    async def embed(self, text: str, task_type: EmbeddingTaskType) -> list[float]:
        raise AIError("embedding backend unavailable")


@pytest.mark.parametrize(
    ("model", "raw_query"),
    [
        (None, "optics"),
        (_FailingEmbeddingModel(), "optics"),
        (FakeEmbeddingModel(dimensions=3, model_id="fake-v1"), "   "),
        (FakeEmbeddingModel(dimensions=3, model_id="fake-v1"), "###"),
    ],
)
async def test_embedding_searcher_embed_query_guards_and_ai_error(
    session_manager: DatabaseSessionManager,
    model: EmbeddingModel | None,
    raw_query: str,
) -> None:
    searcher = EmbeddingChunkSearcher(model)
    query_unit = await searcher.embed_query(raw_query)
    assert query_unit is None
    async with session_manager.session() as session:
        assert (
            await searcher.search(
                session, SearchQuery(user_id=1, raw_query=raw_query), query_unit
            )
            == []
        )


@pytest.mark.parametrize(
    "bad_embedding",
    [
        "",
        "{not-json",
        json.dumps([1.0, 0.0]),
        json.dumps([0.0, 0.0, 0.0]),
    ],
)
async def test_embedding_searcher_skips_corrupt_or_zero_norm_embeddings(
    session_manager: DatabaseSessionManager,
    bad_embedding: str,
) -> None:
    model = FakeEmbeddingModel(dimensions=3, model_id="fake-v1")
    searcher = EmbeddingChunkSearcher(model)
    query_unit = await searcher.embed_query("optics")
    assert query_unit is not None

    async with session_manager.session() as session:
        session.add_all(
            [
                UserFileDO(
                    id=101,
                    user_id=1,
                    file_name="Corrupt.note",
                    directory_id=0,
                    is_active="Y",
                    is_folder="N",
                ),
                NotePageContentDO(
                    file_id=101,
                    page_index=0,
                    page_id="P20260401090000p0",
                    text_content="Unusable vector",
                    embedding=bad_embedding,
                    embedding_model="fake-v1",
                ),
            ]
        )
        await session.commit()
        assert (
            await searcher.search(
                session, SearchQuery(user_id=1, raw_query="optics"), query_unit
            )
            == []
        )


async def test_embedding_searcher_enforces_similarity_floor_and_honors_top_n(
    session_manager: DatabaseSessionManager,
) -> None:
    model = FakeEmbeddingModel(dimensions=3, model_id="fake-v1")
    searcher = EmbeddingChunkSearcher(model)
    query_unit = await searcher.embed_query("🚀")
    assert query_unit is not None
    orth_raw = [float(query_unit[1]), float(-query_unit[0]), 0.0]
    orth_norm = sum(v * v for v in orth_raw) ** 0.5
    orth_unit = np.asarray([v / orth_norm for v in orth_raw], dtype=float)

    async with session_manager.session() as session:
        session.add_all(
            [
                UserFileDO(
                    id=101,
                    user_id=1,
                    file_name="100%_Vectors.note",
                    directory_id=0,
                    is_active="Y",
                    is_folder="N",
                ),
                NotePageContentDO(
                    file_id=101,
                    page_index=0,
                    page_id="P20260401090000p0",
                    text_content="Below similarity floor",
                    embedding=json.dumps((-query_unit).tolist()),
                    embedding_model="fake-v1",
                ),
                NotePageContentDO(
                    file_id=101,
                    page_index=1,
                    page_id="P20260301090000p1",
                    text_content="Before after_date",
                    embedding=json.dumps(query_unit.tolist()),
                    embedding_model="fake-v1",
                ),
                NotePageContentDO(
                    file_id=101,
                    page_index=2,
                    page_id="P20260401090000p2",
                    text_content="Rocket propulsion calculations.",
                    embedding=json.dumps(query_unit.tolist()),
                    embedding_model="fake-v1",
                ),
            ]
        )
        await session.commit()

        hits = await searcher.search(
            session,
            SearchQuery(
                user_id=1,
                raw_query="🚀",
                top_n=1,
                name_filter="100%_",
                after_date=date(2026, 4, 1),
                before_date=date(2026, 4, 1),
            ),
            query_unit,
        )
        all_below_floor = await searcher.search(
            session,
            SearchQuery(user_id=1, raw_query="🚀", top_n=5),
            orth_unit,
        )
        no_candidates = await searcher.search(
            session,
            SearchQuery(user_id=999, raw_query="🚀", top_n=5),
            query_unit,
        )

    assert all_below_floor == []
    assert no_candidates == []
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.0)
    assert hits == [
        SearchResult(
            file_id=101,
            file_name="100%_Vectors.note",
            page_index=2,
            page_id="P20260401090000p2",
            score=hits[0].score,
            text_preview="Rocket propulsion calculations.",
            date="2026-04-01",
        )
    ]


def test_rank_and_merge_results_drops_below_floor_noise_and_preserves_tag_preview() -> (
    None
):
    kw_hits = [
        SearchResult(1, "Neg.note", 0, "p_neg", 1.0, "Keywords: #optics"),
        SearchResult(2, "Hybrid.note", 0, "p_hyb", 0.92, "[#optics] Mirror grinding"),
        SearchResult(3, "TagOnly.note", 0, "", 0.80, "Keywords: #optics"),
    ]
    vec_hits = [
        SearchResult(1, "Neg.note", 0, "p_neg", -0.4, "Unrelated grocery list"),
        SearchResult(2, "Hybrid.note", 0, "p_hyb", 0.70, "Mirror grinding"),
        SearchResult(3, "TagOnly.note", 0, "", 0.60, "Collimation bench notes"),
        SearchResult(4, "Noise.note", 0, "p_noise", 0.05, "Low cosine noise"),
        SearchResult(5, "Semantic.note", 0, "p_sem", 0.55, "Parabolic lens notes"),
    ]

    merged = rank_and_merge_results(kw_hits, vec_hits, top_n=10)
    assert [(m.file_id, m.text_preview) for m in merged] == [
        (1, "Keywords: #optics"),
        (2, "[#optics] Mirror grinding"),
        (3, "Collimation bench notes"),
        (5, "Parabolic lens notes"),
    ]
    assert merged[1].score == pytest.approx(0.92 + 0.5 * 0.70 * 0.08)
    assert rank_and_merge_results(kw_hits, vec_hits, top_n=0) == []
