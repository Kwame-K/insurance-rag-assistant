"""Download the public source documents listed in data/sources.yaml.

Usage:
    uv run python scripts/fetch_corpus.py              # download everything missing
    uv run python scripts/fetch_corpus.py --only osfi_e21_2024
    uv run python scripts/fetch_corpus.py --force      # re-download and refresh hashes

Files land in data/external/raw/ (git-ignored). A manifest with URL, retrieval
date and SHA-256 is written to data/external/download_manifest.json.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES_FILE = ROOT / "data" / "sources.yaml"
OUTPUT_DIR = ROOT / "data" / "external" / "raw"
MANIFEST_FILE = ROOT / "data" / "external" / "download_manifest.json"

# Replace the contact with your own so site operators can reach you.
USER_AGENT = "insurance-rag-assistant/0.1 (personal research project; contact: kristian.klaban@gmail.com)"
CONTENT_TYPES = {"html": "text/html", "pdf": "application/pdf"}
REQUEST_DELAY_SECONDS = 2.0
MAX_ATTEMPTS = 3


def load_sources() -> list[dict[str, Any]]:
    data = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        message = f"{SOURCES_FILE} must contain a YAML list."
        raise TypeError(message)
    return data


def load_manifest() -> dict[str, Any]:
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    return {}


def fetch(client: httpx.Client, url: str) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.get(url)
            if response.status_code == 429 or response.status_code >= 500:
                message = f"HTTP {response.status_code}"
                raise httpx.HTTPError(message)
            return response
        except httpx.HTTPError as error:
            last_error = error
            time.sleep(2**attempt)
    message = f"Failed after {MAX_ATTEMPTS} attempts: {last_error}"
    raise RuntimeError(message)


def validate(source: dict[str, Any], response: httpx.Response) -> None:
    """Raise ValueError if the response is not the document we expect."""
    source_id = source["id"]
    fmt = source["format"]
    if response.status_code != 200:
        message = f"{source_id}: HTTP {response.status_code}"
        raise ValueError(message)

    content_type = response.headers.get("content-type", "").lower()
    if CONTENT_TYPES[fmt] not in content_type:
        message = f"{source_id}: expected {CONTENT_TYPES[fmt]}, got {content_type!r}"
        raise ValueError(message)

    min_bytes = int(source.get("min_bytes", 10_000))
    if len(response.content) < min_bytes:
        message = (
            f"{source_id}: only {len(response.content)} bytes (minimum {min_bytes})"
        )
        raise ValueError(message)

    if fmt == "pdf" and not response.content.startswith(b"%PDF"):
        message = f"{source_id}: response is not a PDF file"
        raise ValueError(message)

    if fmt == "html":
        text = response.text.lower()
        missing = [t for t in source.get("must_contain", []) if t.lower() not in text]
        if missing:
            message = f"{source_id}: expected text not found: {missing}"
            raise ValueError(message)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", action="append", default=[], help="Source id (repeatable)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Download even if the file exists."
    )
    args = parser.parse_args()

    sources = load_sources()
    if args.only:
        sources = [s for s in sources if s["id"] in args.only]
        if not sources:
            print(f"No source matches {args.only}", file=sys.stderr)
            return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    failures = 0

    with httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en,fr;q=0.8"},
        follow_redirects=True,
        timeout=httpx.Timeout(60.0),
    ) as client:
        for index, source in enumerate(sources):
            source_id = source["id"]
            destination = OUTPUT_DIR / f"{source_id}.{source['format']}"

            if destination.exists() and not args.force:
                print(f"SKIP   {source_id} (already downloaded)")
                continue

            if index > 0:
                time.sleep(REQUEST_DELAY_SECONDS)

            try:
                response = fetch(client, source["url"])
                validate(source, response)
            except (RuntimeError, ValueError) as error:
                failures += 1
                print(f"FAIL   {error}", file=sys.stderr)
                continue

            destination.write_bytes(response.content)
            manifest[source_id] = {
                "url": source["url"],
                "final_url": str(response.url),
                "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "sha256": sha256_of(destination),
                "bytes": len(response.content),
                "content_type": response.headers.get("content-type"),
                "license_note": source.get("license_note"),
                "redistribute": bool(source.get("redistribute", False)),
            }
            print(f"OK     {source_id} ({len(response.content):,} bytes)")

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
