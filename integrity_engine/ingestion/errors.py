class ExtractionError(Exception):
    """Raised when a file cannot yield usable text. The pipeline flags and skips it."""
