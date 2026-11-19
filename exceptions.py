"""Custom exceptions used across the language model management system."""


class InvalidPromptError(Exception):
    """Raised when a prompt is empty, whitespace only, or not a string."""


class ModelNotFoundError(Exception):
    """Raised when a requested model does not exist in the ModelManager."""
