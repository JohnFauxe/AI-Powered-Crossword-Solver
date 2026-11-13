"""Download SCOWL and build ``$CROSSWORD_DATA/words.txt``, the solver's word list.

This is one of the only pieces of the project allowed to use the network, and
only when you run it::

    uv run python scripts/build_wordlist.py            # download + build
    uv run python scripts/build_wordlist.py --rebuild  # overwrite words.txt

SCOWL (Spell Checker Oriented Word Lists, Kevin Atkinson) is the source of the
Debian/Ubuntu ``wamerican`` and ``wbritish`` packages. This script takes the
same lists those packages use (American and British spellings and their
variants, size 50 and below, words, proper names, abbreviations and
contractions), so the result is a superset of ``/usr/share/dict/american-english``
and ``british-english`` on any OS. The download is pinned and checked against
its SHA-256 (about 2.5 MB). SCOWL's licence permits redistribution with its
copyright notice; the notice is saved next to the list as ``words-LICENSE.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import tarfile
import time

import httpx

from crossword.store import data_dir

VERSION = "2020.12.07"
URL = (
    f"https://downloads.sourceforge.net/project/wordlist/SCOWL/{VERSION}/scowl-{VERSION}.tar.gz"
)
SHA256 = "5587667caa20c4891390c2d42dbb4d5c4c3f41bee77af1457ece3ba23fb859cc"
MAX_SIZE = 50
# Spelling regions taken from final/<region>-<kind>.<size>; "special" covers
# special-hacker and special-roman-numerals, as the Debian packages do.
REGIONS = {
    "english", "american", "british", "variant_1", "variant_2",
    "british_variant_1", "british_variant_2", "special",
}
FINAL = re.compile(r"^scowl-[\d.]+/final/(?P<region>[a-z_0-9]+)-[a-z-]+\.(?P<size>\d+)$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download() -> bytes:
    log(f"downloading SCOWL {VERSION}")
    r = httpx.get(URL, follow_redirects=True, timeout=120)
    r.raise_for_status()
    digest = hashlib.sha256(r.content).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: got {digest}, expected {SHA256}")
    log(f"downloaded {len(r.content) / 1e6:.1f} MB, checksum ok")
    return r.content


def extract(archive: bytes) -> tuple[list[str], str]:
    """The selected words (sorted, deduplicated) and the licence text."""
    words: set[str] = set()
    licence = ""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name.endswith("/Copyright"):
                licence = tar.extractfile(member).read().decode("latin-1")
                continue
            m = FINAL.match(member.name)
            if not m or m["region"] not in REGIONS or int(m["size"]) > MAX_SIZE:
                continue
            text = tar.extractfile(member).read().decode("latin-1")
            words.update(line.strip() for line in text.splitlines() if line.strip())
    return sorted(words), licence


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true", help="overwrite an existing words.txt")
    args = ap.parse_args(argv)

    out = data_dir() / "words.txt"
    if out.exists() and not args.rebuild:
        log(f"{out} already exists (use --rebuild to overwrite)")
        return 0
    words, licence = extract(download())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".txt.part")
    tmp.write_text("".join(w + "\n" for w in words), encoding="utf-8")
    tmp.replace(out)
    (out.parent / "words-LICENSE.txt").write_text(licence, encoding="utf-8")
    log(f"wrote {len(words):,} words to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
