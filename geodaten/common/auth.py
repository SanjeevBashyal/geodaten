"""Resolve only user-owned credentials without persisting or logging them."""

from __future__ import annotations

import os
from contextlib import contextmanager
from threading import RLock
from typing import Iterator, Mapping

from geodaten.common.errors import AuthenticationError
from geodaten.common.models import CredentialField

_ENVIRONMENT_LOCK = RLock()


def resolve_credentials(
    fields: tuple[CredentialField, ...],
    supplied: Mapping[str, str] | None = None,
) -> dict[str, str]:
    supplied = supplied or {}
    resolved: dict[str, str] = {}
    for field in fields:
        raw = supplied.get(field.id)
        value = raw if isinstance(raw, str) else (str(raw) if raw is not None else "")
        if not value.strip():
            for name in field.environment:
                environment_value = os.getenv(name)
                value = environment_value if environment_value is not None else ""
                if value.strip():
                    break
        if value.strip():
            resolved[field.id] = value
    return resolved


def require_credentials(
    fields: tuple[CredentialField, ...],
    supplied: Mapping[str, str] | None = None,
) -> dict[str, str]:
    resolved = resolve_credentials(fields, supplied)
    missing = [field.title for field in fields if field.required and not resolved.get(field.id)]
    if missing:
        raise AuthenticationError("Missing user credentials: " + ", ".join(missing))
    return resolved


def redacted(values: Mapping[str, str]) -> dict[str, str]:
    return {key: "***" if value else "" for key, value in values.items()}


@contextmanager
def scoped_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Temporarily expose credentials to a library that only supports env vars."""

    with _ENVIRONMENT_LOCK:
        previous = {key: os.environ.get(key) for key in values}
        os.environ.update({key: value for key, value in values.items() if value})
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def child_environment(values: Mapping[str, str]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({key: value for key, value in values.items() if value})
    return environment
