"""Shared error type for all formats."""


class FormatError(ValueError):
    """The input cannot be read as a puzzle, or the puzzle cannot be written
    in the requested format. The message says why, in plain words."""
