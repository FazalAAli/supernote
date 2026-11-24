"""LLM Wiki server component (`WikiComponent`).

Owns per-user wiki storage (`WikiService`), Obsidian Markdown/ZIP export
(`WikiExportService`), and the incremental event-driven compiler
(`WikiCompilerWorker`). Registered opt-in (`default_enabled=False`) in
`BUILTIN_COMPONENTS`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from supernote.server.components.ai import TEXT_MODEL_KEY
from supernote.server.components.events.service import EVENT_LOG_SERVICE_KEY
from supernote.server.core import ServerCore

from .compiler import WIKI_COMPILER_KEY, WikiCompilerWorker
from .config import WikiConfig
from .export import WIKI_EXPORT_SERVICE_KEY, WikiExportService
from .service import WIKI_SERVICE_KEY, WikiService

__all__ = [
    "WIKI_COMPILER_KEY",
    "WIKI_EXPORT_SERVICE_KEY",
    "WIKI_SERVICE_KEY",
    "WikiCompilerWorker",
    "WikiComponent",
    "WikiConfig",
    "WikiExportService",
    "WikiService",
]


class WikiComponent:
    """Vertical component managing per-user LLM wiki storage, export, and compilation."""

    domain = "wiki"
    dependencies: tuple[str, ...] = ("ai", "events")
    platforms: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._config: WikiConfig | None = None
        self._compiler: WikiCompilerWorker | None = None

    @property
    def config(self) -> WikiConfig | None:
        """Validated `WikiConfig` after `async_setup`."""
        return self._config

    async def async_setup(self, core: ServerCore, config: Mapping[str, Any]) -> None:
        """Validate `WikiConfig`, instantiate services, and expose app keys."""
        self._config = WikiConfig.from_dict(config)
        service = WikiService(core.session_manager)
        compiler = WikiCompilerWorker(
            config=self._config,
            session_manager=core.session_manager,
            wiki_service=service,
            event_log_service=core.app[EVENT_LOG_SERVICE_KEY],
            event_bus=core.event_bus,
            text_model=core.app.get(TEXT_MODEL_KEY),
        )
        self._compiler = compiler
        core.app[WIKI_SERVICE_KEY] = service
        core.app[WIKI_EXPORT_SERVICE_KEY] = WikiExportService(service)
        core.app[WIKI_COMPILER_KEY] = compiler

    async def async_start(self, core: ServerCore) -> None:
        """Bind finalized `TEXT_MODEL_KEY` and start event subscriptions."""
        if self._compiler is not None:
            self._compiler.set_text_model(core.app.get(TEXT_MODEL_KEY))
            await self._compiler.start()

    async def async_stop(self, core: ServerCore) -> None:
        """Idempotently stop background debounce timers and release references."""
        compiler = self._compiler
        self._compiler = None
        self._config = None
        if compiler is not None:
            await compiler.stop()
