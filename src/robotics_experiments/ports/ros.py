from __future__ import annotations

from typing import Protocol


class TogglePublisher(Protocol):
    def publish_toggle(self, index: int, state: bool) -> None: ...
