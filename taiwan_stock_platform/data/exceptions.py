class DataSourceError(RuntimeError):
    """Raised when an upstream data source fails."""


class DataNotFoundError(DataSourceError):
    """Raised when a data source returns no usable rows."""

