"""Unit tests for `supernote.server.components.search.service.SearchService`."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from supernote.server.ai.models.embedding import EmbeddingTaskType
from supernote.server.components.fake_ai.embedding import FakeEmbeddingModel
from supernote.server.components.processor.page_hashing import PageHashingModule
from supernote.server.components.search import (
    NotebookInfo,
    SearchResult,
    SearchService,
)
from supernote.server.constants import USER_DATA_BUCKET
from supernote.server.db.models.file import UserFileDO
from supernote.server.db.models.note_processing import (
    NotePageContentDO,
    NotePageMetadataDO,
)
from supernote.server.db.session import DatabaseSessionManager
from supernote.server.services.file import FileService
from supernote.testing.notebook_fixtures import build_synthetic_notebook


@pytest.fixture
def fake_embedding_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture
def search_service(
    session_manager: DatabaseSessionManager,
    fake_embedding_model: FakeEmbeddingModel,
) -> SearchService:
    return SearchService(
        session_manager=session_manager,
        embedding_model=fake_embedding_model,
    )


async def test_search_service_late_bound_embedding_model_adds_semantic_hits(
    session_manager: DatabaseSessionManager,
    fake_embedding_model: FakeEmbeddingModel,
) -> None:
    """Keyword search works without embeddings, and late-binding `embedding_model` adds semantic hits."""
    service = SearchService(session_manager=session_manager, embedding_model=None)
    assert service.embedding_model is None

    feline_query_vec = await fake_embedding_model.embed(
        "feline companions", task_type=EmbeddingTaskType.RETRIEVAL_QUERY
    )
    async with session_manager.session() as session:
        session.add_all(
            [
                UserFileDO(
                    id=101,
                    user_id=1,
                    file_name="Pets.note",
                    directory_id=0,
                    is_active="Y",
                    is_folder="N",
                ),
                NotePageMetadataDO(
                    user_id=1,
                    file_id=101,
                    page_index=0,
                    page_id="P20260401090000a",
                    keywords_json=json.dumps(
                        [{"KEYWORDPAGE": "1", "KEYWORD": "vet-visit"}]
                    ),
                ),
                NotePageContentDO(
                    file_id=101,
                    page_index=1,
                    page_id="P20260402090000b",
                    text_content="Cats love sunny windowsills.",
                    embedding=json.dumps(feline_query_vec),
                    embedding_model=fake_embedding_model.model_id,
                ),
            ]
        )
        await session.commit()

    tag_hits = await service.search_chunks(user_id=1, query="#vet-visit")
    assert [h.page_index for h in tag_hits] == [0]
    assert await service.search_chunks(user_id=1, query="feline companions") == []

    service.embedding_model = fake_embedding_model
    assert service.embedding_model is fake_embedding_model

    sem_hits = await service.search_chunks(user_id=1, query="feline companions")
    assert [h.page_index for h in sem_hits] == [1]


async def test_list_notebooks_escapes_sql_wildcards_and_orders_active_notes(
    search_service: SearchService,
    session_manager: DatabaseSessionManager,
) -> None:
    async with session_manager.session() as session:
        session.add_all(
            [
                UserFileDO(
                    id=201,
                    user_id=1,
                    file_name="100%_Summary.note",
                    directory_id=0,
                    is_folder="N",
                    is_active="Y",
                    create_time=1000,
                    update_time=3000,
                ),
                UserFileDO(
                    id=202,
                    user_id=1,
                    file_name="1000XSummary.note",
                    directory_id=0,
                    is_folder="N",
                    is_active="Y",
                    create_time=1000,
                    update_time=2000,
                ),
                UserFileDO(
                    id=203,
                    user_id=1,
                    file_name="100%_Deleted.note",
                    directory_id=0,
                    is_folder="N",
                    is_active="N",
                    create_time=1000,
                    update_time=9000,
                ),
            ]
        )
        await session.commit()

    all_notes = await search_service.list_notebooks(user_id=1)
    assert all_notes == [
        NotebookInfo(
            file_id=201,
            file_name="100%_Summary.note",
            create_time=1000,
            update_time=3000,
        ),
        NotebookInfo(
            file_id=202,
            file_name="1000XSummary.note",
            create_time=1000,
            update_time=2000,
        ),
    ]

    escaped = await search_service.list_notebooks(user_id=1, name_filter="100%_")
    assert [n.file_name for n in escaped] == ["100%_Summary.note"]


@pytest.mark.parametrize(
    ("query", "top_n", "date_after", "date_before"),
    [
        ("   ", 5, None, None),
        ("###", 5, None, None),
        ("🚀", 5, None, None),
        ("cats", 0, None, None),
        ("cats", -1, None, None),
        ("cats", 5, "not-a-date", None),
        ("cats", 5, "2026-05-01", "2026-04-01"),
    ],
)
async def test_search_chunks_guard_clauses(
    search_service: SearchService,
    query: str,
    top_n: int,
    date_after: str | None,
    date_before: str | None,
) -> None:
    assert (
        await search_service.search_chunks(
            user_id=1,
            query=query,
            top_n=top_n,
            date_after=date_after,
            date_before=date_before,
        )
        == []
    )


async def test_hybrid_search_end_to_end_with_synthetic_notebooks_and_noise_floor(
    session_manager: DatabaseSessionManager,
    file_service: FileService,
) -> None:
    """End-to-end hybrid search ranks hybrid > tag-only > semantic and drops unrelated noise."""
    user_id = 1
    hashing_module = PageHashingModule(file_service=file_service)
    token_model = FakeEmbeddingModel(dimensions=128, blended=True)
    service = SearchService(
        session_manager=session_manager, embedding_model=token_model
    )

    unprocessed_nb = build_synthetic_notebook(
        pages=["# Optics Sketch\n\n- Collimation diagram #optics #astronomy"],
        base_date=datetime(2026, 3, 10, 9, 0, tzinfo=UTC),
        seed=11,
    )
    processed_nb = build_synthetic_notebook(
        pages=[
            "# Mirror Grinding\n\n- Parabolic mirror focal ratio and lens aberration #optics",
            "# Star Party Log\n\n- Checked lens aberration on Jupiter moons",
            "# Grocery Checklist\n\n- Buy laundry detergent, bananas, and dish soap",
        ],
        base_date=datetime(2026, 3, 12, 9, 0, tzinfo=UTC),
        seed=22,
    )

    for fid, fname, synth in [
        (401, "Unprocessed_Sketch.note", unprocessed_nb),
        (402, "Optics_Lab.note", processed_nb),
    ]:
        skey = f"hybrid-{fid}.note"
        await file_service.blob_storage.put(USER_DATA_BUCKET, skey, synth.data)
        async with session_manager.session() as session:
            session.add(
                UserFileDO(
                    id=fid,
                    user_id=user_id,
                    directory_id=0,
                    file_name=fname,
                    is_folder="N",
                    is_active="Y",
                    storage_key=skey,
                    md5=hashlib.md5(synth.data).hexdigest(),
                    size=len(synth.data),
                )
            )
            await session.commit()
        await hashing_module.process(fid, session_manager)

    async with session_manager.session() as session:
        for page in processed_nb.pages:
            vec = await token_model.embed(
                page.text, task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT
            )
            content_row = (
                await session.execute(
                    select(NotePageContentDO).where(
                        NotePageContentDO.file_id == 402,
                        NotePageContentDO.page_id == page.page_id,
                    )
                )
            ).scalar_one()
            content_row.text_content = page.text
            content_row.embedding = json.dumps(vec)
            content_row.embedding_model = token_model.model_id
        await session.commit()

    ranked = await service.search_chunks(
        user_id=user_id, query="#optics aberration", top_n=5
    )
    assert len(ranked) == 3
    assert ranked[0].score > ranked[1].score > ranked[2].score >= 0.25
    assert ranked == [
        SearchResult(
            file_id=402,
            file_name="Optics_Lab.note",
            page_index=0,
            page_id=processed_nb.pages[0].page_id,
            score=ranked[0].score,
            text_preview=processed_nb.pages[0].text.strip(),
            date="2026-03-12",
        ),
        SearchResult(
            file_id=401,
            file_name="Unprocessed_Sketch.note",
            page_index=0,
            page_id=unprocessed_nb.pages[0].page_id,
            score=ranked[1].score,
            text_preview="Keywords: #optics, #astronomy",
            date="2026-03-10",
        ),
        SearchResult(
            file_id=402,
            file_name="Optics_Lab.note",
            page_index=1,
            page_id=processed_nb.pages[1].page_id,
            score=ranked[2].score,
            text_preview=processed_nb.pages[1].text.strip(),
            date="2026-03-13",
        ),
    ]
