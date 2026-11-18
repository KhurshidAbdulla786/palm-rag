class AppError(Exception):
    """Base class for errors that map cleanly onto an HTTP status code."""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(AppError):
    status_code = 415


class FileTooLargeError(AppError):
    status_code = 413


class TextExtractionError(AppError):
    status_code = 422


class EmptyDocumentError(AppError):
    status_code = 422


class InvalidChunkingConfigError(AppError):
    status_code = 422
