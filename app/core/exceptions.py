"""Custom application exceptions following Clean Architecture conventions."""


class AppException(Exception):
    """Base exception for all application-level errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class NotFoundError(AppException):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource: str, identifier: str | int) -> None:
        super().__init__(
            message=f"{resource} with identifier '{identifier}' was not found.",
            status_code=404,
        )


class ConflictError(AppException):
    """Raised when an operation conflicts with existing state (e.g. duplicate)."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=409)


class ValidationError(AppException):
    """Raised when input data fails business-rule validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=422)


class UnauthorizedError(AppException):
    """Raised when authentication or authorisation fails."""

    def __init__(self, message: str = "Authentication required.") -> None:
        super().__init__(message=message, status_code=401)


class InternalServerError(AppException):
    """Raised for unexpected internal errors."""

    def __init__(self, message: str = "An unexpected error occurred.") -> None:
        super().__init__(message=message, status_code=500)
