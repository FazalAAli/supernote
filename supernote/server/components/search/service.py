"""Hybrid keyword + semantic vector search and notebook listing service.

`SearchService` orchestrates `KeywordChunkSearcher` (matching native Supernote
`NotePageMetadataDO` keywords and SQL-prefiltered `NotePageContentDO` transcripts)
alongside `EmbeddingChunkSearcher` (computing cosine similarity via `EmbeddingModel`)
inside a single SQLite read snapshot, merging results via `rank_and_merge_results`.

### Internal Component Access

Other server components or plugins declaring ``dependencies = ("search",)``
retrieve the shared service instance via `core.app[SEARCH_SERVICE_KEY]`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from aiohttp import web
from sqlalchemy import select

from supernote.server.ai.models.embedding import EmbeddingModel
from supernote.server.db.models.file import UserFileDO
from supernote.server.db.session import DatabaseSessionManager

from .chunk_search import (
    EmbeddingChunkSearcher,
    KeywordChunkSearcher,
    SearchQuery,
    rank_and_merge_results,
)
from .lexical_scoring import parse_lexical_query
from .models import NotebookInfo, SearchResult

__all__ = [
    "SEARCH_SERVICE_KEY",
    "SearchService",
]

logger = logging.getLogger(__name__)


class SearchService:
    """Service for hybrid keyword + semantic search and active notebook listing.

    All queries are strictly scoped by `user_id` and restricted to active
    (`UserFileDO.is_active == "Y"`) `.note` files. When `embedding_model` is
    `None`, `search_chunks` still executes lexical `#keyword` and transcript
    search via `KeywordChunkSearcher`.
    """

    def __init__(
        self,
        session_manager: DatabaseSessionManager,
        embedding_model: EmbeddingModel | None,
    ) -> None:
        """Initialize the search service.

        Args:
            session_manager: Async SQLAlchemy session manager for querying
                `UserFileDO`, `NotePageMetadataDO`, and `NotePageContentDO`.
            embedding_model: Optional configured embedding backend used to
                embed search queries. May be bound later during startup once
                AI capability platforms finish loading.
        """
        self.session_manager = session_manager
        self._keyword_searcher = KeywordChunkSearcher()
        self._embedding_searcher = EmbeddingChunkSearcher(embedding_model)

    @property
    def embedding_model(self) -> EmbeddingModel | None:
        """Configured embedding backend used to embed search queries."""
        return self._embedding_searcher.embedding_model

    @embedding_model.setter
    def embedding_model(self, model: EmbeddingModel | None) -> None:
        self._embedding_searcher.embedding_model = model

    async def list_notebooks(
        self,
        user_id: int,
        name_filter: str | None = None,
        limit: int = 20,
    ) -> list[NotebookInfo]:
        """List active `.note` notebooks for a user ordered by most recently updated."""
        async with self.session_manager.session() as session:
            stmt = (
                select(UserFileDO)
                .where(UserFileDO.user_id == user_id)
                .where(UserFileDO.is_active == "Y")
                .where(UserFileDO.is_folder == "N")
                .where(UserFileDO.file_name.ilike("%.note"))
            )
            if name_filter:
                stmt = stmt.where(
                    UserFileDO.file_name.icontains(name_filter, autoescape=True)
                )
            stmt = stmt.order_by(UserFileDO.update_time.desc()).limit(limit)
            files = (await session.execute(stmt)).scalars().all()

        return [
            NotebookInfo(
                file_id=f.id,
                file_name=f.file_name,
                create_time=f.create_time,
                update_time=f.update_time,
            )
            for f in files
        ]

    async def search_chunks(
        self,
        user_id: int,
        query: str,
        top_n: int = 5,
        name_filter: str | None = None,
        date_after: str | None = None,
        date_before: str | None = None,
    ) -> list[SearchResult]:
        """Search active `.note` pages by `#keyword` tags, OCR text, and vector similarity.

        Args:
            user_id: Owner user ID performing the search.
            query: Keyword (`#tag`), phrase, or semantic search query string.
            top_n: Maximum number of ranked results to return.
            name_filter: Optional case-insensitive substring filter on notebook
                filenames.
            date_after: Optional ISO date string (`YYYY-MM-DD`). Keeps pages
                whose inferred date is on or after this date.
            date_before: Optional ISO date string (`YYYY-MM-DD`). Keeps pages
                whose inferred date is on or before this date.

        Returns:
            List of `SearchResult` sorted by descending hybrid relevance score.
        """
        if top_n <= 0 or not query or not query.strip():
            return []

        after_date: date | None = None
        before_date: date | None = None
        try:
            if date_after:
                after_date = datetime.strptime(date_after, "%Y-%m-%d").date()
            if date_before:
                before_date = datetime.strptime(date_before, "%Y-%m-%d").date()
        except ValueError as e:
            logger.error(f"Invalid date filter format: {e}")
            return []

        if after_date and before_date and after_date > before_date:
            return []

        query_unit = await self._embedding_searcher.embed_query(query)
        if query_unit is None and parse_lexical_query(query) is None:
            return []

        search_query = SearchQuery(
            user_id=user_id,
            raw_query=query,
            top_n=top_n * 2,
            name_filter=name_filter,
            after_date=after_date,
            before_date=before_date,
        )
        async with self.session_manager.session() as session:
            keyword_hits = await self._keyword_searcher.search(session, search_query)
            embedding_hits = await self._embedding_searcher.search(
                session, search_query, query_unit
            )
        return rank_and_merge_results(keyword_hits, embedding_hits, top_n=top_n)


SEARCH_SERVICE_KEY: web.AppKey[SearchService] = web.AppKey("search_service")
"""Typed `aiohttp.web.AppKey` exposing `SearchService` to sibling components and routes."""
