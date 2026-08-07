"""User-facing exception hierarchy."""


class GeodatenError(RuntimeError):
    """Base class for expected failures."""


class ConfigurationError(GeodatenError):
    """A request or provider option is invalid."""


class AuthenticationError(GeodatenError):
    """Required user-owned credentials are unavailable or invalid."""


class DependencyError(GeodatenError):
    """An optional Python package or external program is missing."""


class DownloadError(GeodatenError):
    """A source asset could not be discovered, transferred, or validated."""


class OutputError(GeodatenError):
    """A requested output could not be produced or validated."""


class CancelledError(GeodatenError):
    """The caller cancelled an active operation."""
