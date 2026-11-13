"""Download everything the solver needs: the word list, then the clue database.

One of the only pieces of the project allowed to use the network, and only when
you run it::

    uv run python scripts/setup_data.py              # word list + clue DB
    uv run python scripts/setup_data.py --no-cluedb  # word list only (skip 1.1 GB)

Runs ``build_wordlist.py`` (SCOWL, about 2.5 MB) and ``build_cluedb.py``
(CrosswordQA, about 1.1 GB on disk). Both skip work already done, so it is
safe to run again; run either script on its own for its other options.
"""

from __future__ import annotations

import argparse
import sys

import build_cluedb
import build_wordlist


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--no-cluedb", action="store_true", help="skip the clue database (Lookup mode)"
    )
    args = ap.parse_args(argv)

    rc = build_wordlist.main([])
    if rc or args.no_cluedb:
        return rc
    return build_cluedb.main([])


if __name__ == "__main__":
    sys.exit(main())
