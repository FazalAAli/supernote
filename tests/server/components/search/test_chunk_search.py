"""Unit tests for `supernote.server.components.search.chunk_search`."""

from __future__ import annotations

import json
from datetime import date

import pytest

from supernote.server.ai.base import AIError
from supernote.server.ai.models.embedding import EmbeddingModel, EmbeddingTaskType
from supernote.server.components.fake_ai.embedding import FakeEmbeddingModel
from supernote.server.components.search.chunk_search import (
    EmbeddingChunkSearcher,
    SearchQuery,
)
from supernote.server.components.search.models import SearchResult
from supernote.server.db.models.file import UserFileDO
from supernote.server.db.models.note_processing import NotePageContentDO
from supernote.server.db.session import DatabaseSessionManager


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
        (FakeEmbeddingModel(dimensions=3, model_id="fake-v1"), "---"),
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


async def test_embedding_searcher_filters_note_files_dates_and_escaped_wildcards(
    session_manager: DatabaseSessionManager,
) -> None:
    model = FakeEmbeddingModel(dimensions=3, model_id="fake-v1")
    searcher = EmbeddingChunkSearcher(model)
    query_unit = await searcher.embed_query("🚀")
    assert query_unit is not None

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
                UserFileDO(
                    id=102,
                    user_id=1,
                    file_name="100%_Document.pdf",
                    directory_id=0,
                    is_active="Y",
                    is_folder="N",
                ),
                NotePageContentDO(
                    file_id=101,
                    page_index=0,
                    page_id="P20260402090000p0",
                    text_content="After before_date",
                    embedding=json.dumps(query_unit.tolist()),
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
                NotePageContentDO(
                    file_id=102,
                    page_index=0,
                    page_id="P20260401090000pdf",
                    text_content="Non-note file should be ignored.",
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
        other_user_hits = await searcher.search(
            session,
            SearchQuery(user_id=999, raw_query="🚀", top_n=5),
            query_unit,
        )

    assert other_user_hits == []
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
