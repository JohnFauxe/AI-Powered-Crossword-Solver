"""Read and write puzzles in ``.puz``, ``.ipuz`` and native JSON.

Public API::

    load(path) -> Puzzle                       # by extension, else by content
    loads(data, filename=None) -> Puzzle       # bytes from an upload, say
    dump(puzzle, path, fmt=None) -> None       # by extension unless fmt given
    dumps(puzzle, fmt) -> bytes
    detect_format(data, filename=None) -> "puz" | "ipuz" | "native"
    FormatError

All raise ``FormatError`` (a ``ValueError``) with a plain message when the
input cannot be read or the puzzle cannot be written in that format. Reading
never validates; call ``Puzzle.validate()`` for that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from crossword.model import Puzzle

from . import ipuz, native, puz
from .errors import FormatError

Format = Literal["puz", "ipuz", "native"]
FORMATS: tuple[Format, ...] = ("puz", "ipuz", "native")

_EXTENSIONS: dict[str, Format] = {".puz": "puz", ".ipuz": "ipuz", ".json": "native"}

__all__ = [
    "FORMATS",
    "Format",
    "FormatError",
    "detect_format",
    "dump",
    "dumps",
    "format_for_path",
    "load",
    "loads",
]


def format_for_path(path: str | Path) -> Format | None:
    """Format implied by a file name's extension, or None if unknown."""
    return _EXTENSIONS.get(Path(path).suffix.lower())


def detect_format(data: bytes, filename: str | None = None) -> Format:
    """Decide the format from content first, falling back to the extension."""
    if b"ACROSS&DOWN\0" in data[:4096]:
        return "puz"
    text = data.lstrip(b"\xef\xbb\xbf \t\r\n")
    if text.startswith(b"ipuz("):
        return "ipuz"
    if text.startswith(b"{"):
        try:
            doc = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            doc = None
        if isinstance(doc, dict):
            if "kind" in doc or "dimensions" in doc or "puzzle" in doc:
                return "ipuz"
            if "grid" in doc:
                return "native"
    by_ext = format_for_path(filename) if filename else None
    if by_ext:
        return by_ext
    raise FormatError("could not tell the file format; expected .puz, .ipuz or native JSON")


def loads(data: bytes, filename: str | None = None) -> Puzzle:
    """Parse puzzle bytes. ``filename`` only helps format detection."""
    fmt = detect_format(data, filename)
    if fmt == "puz":
        return puz.read(data)
    if fmt == "ipuz":
        return ipuz.read(data)
    return native.read(data)


def load(path: str | Path) -> Puzzle:
    """Read a puzzle file."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as e:
        raise FormatError(f"cannot read {path}: {e.strerror}") from e
    return loads(data, path.name)


def dumps(puzzle: Puzzle, fmt: Format) -> bytes:
    if fmt == "puz":
        return puz.write(puzzle)
    if fmt == "ipuz":
        return ipuz.write(puzzle)
    if fmt == "native":
        return native.write(puzzle)
    raise FormatError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")


def dump(puzzle: Puzzle, path: str | Path, fmt: Format | None = None) -> None:
    """Write a puzzle; the format comes from ``fmt`` or the extension."""
    path = Path(path)
    fmt = fmt or format_for_path(path)
    if fmt is None:
        raise FormatError(f"cannot tell the format from {path.name!r}; use .puz, .ipuz or .json")
    data = dumps(puzzle, fmt)
    path.write_bytes(data)
