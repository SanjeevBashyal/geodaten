"""Interface-neutral progress and cancellation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Callable

from geodaten.common.errors import CancelledError


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    message: str
    current: int | None = None
    total: int | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None


ProgressCallback = Callable[[ProgressEvent], None]


def null_progress(_event: ProgressEvent) -> None:
    return None


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise CancelledError("Download cancelled")
