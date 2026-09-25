"""Collect raw HN stories and comments via Algolia; runnable directly."""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

SEARCH_URL = "https://hn.algolia.com/api/v1/search"
RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
MAX_PAGES = 5


def search_page(client: httpx.Client, query: str, kind: str, page: int) -> dict:
    """Retry temporary network/server errors, with a bounded delay."""
    for attempt in range(3):
        try:
            response = client.get(
                SEARCH_URL,
                params={"query": query, "tags": kind, "page": page, "hitsPerPage": 30},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list):
                raise ValueError("Unexpected Hacker News search response")
            if not isinstance(payload.get("nbPages"), int):
                raise ValueError("Missing Hacker News pagination metadata")
            return payload
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                exc.response.status_code == 429 or exc.response.status_code >= 500
            )
            if attempt == 2 or not retryable:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Search retry loop exhausted")


def collect(company: str, limit: int = 200, *, client: httpx.Client) -> dict:
    company = company.strip()
    if not company:
        raise ValueError("Company must not be blank")
    if not 1 <= limit <= 300:
        raise ValueError("Limit must be between 1 and 300")
    queries = [company] + [f"{company} {term}" for term in (
        "employees", "layoffs", "management", "promotion"
    )]
    streams = [(query, kind) for query in queries for kind in ("story", "comment")]
    exhausted = set()
    items = {}
    requests_made = 0
    collected_at = datetime.now(timezone.utc).isoformat()

    for page in range(MAX_PAGES):
        batches = []
        for query, kind in streams:
            if (query, kind) in exhausted:
                continue
            payload = search_page(client, query, kind, page)
            requests_made += 1
            batches.append((query, kind, payload["hits"]))
            if page + 1 >= payload["nbPages"] or not payload["hits"]:
                exhausted.add((query, kind))

        # Round robin gives each query and item type a chance before the cap.
        for index in range(max((len(hits) for _, _, hits in batches), default=0)):
            for query, kind, hits in batches:
                if index >= len(hits):
                    continue
                hit = hits[index]
                if not isinstance(hit, dict):
                    raise ValueError("Unexpected Hacker News item")
                item_id = str(hit.get("objectID", ""))
                if not item_id.isdigit() or hit.get("deleted") or hit.get("dead"):
                    continue
                if kind == "comment" and not hit.get("comment_text"):
                    continue
                if kind == "story" and not (hit.get("title") or hit.get("story_text")):
                    continue
                if item_id in items:
                    if query not in items[item_id]["matched_queries"]:
                        items[item_id]["matched_queries"].append(query)
                elif len(items) < limit:
                    items[item_id] = {
                        "id": item_id,
                        "type": kind,
                        "source_url": f"https://news.ycombinator.com/item?id={item_id}",
                        "matched_queries": [query],
                        "raw": hit,
                    }
        if len(items) >= limit or len(exhausted) == len(streams):
            break

    documents = list(items.values())
    return {
        "company": company,
        "source": "hacker_news",
        "source_type": "community",
        "search_api": SEARCH_URL,
        "collected_at": collected_at,
        "queries": queries,
        "requested_limit": limit,
        "item_count": len(documents),
        "counts": {kind: sum(item["type"] == kind for item in documents)
                   for kind in ("story", "comment")},
        "requests_made": requests_made,
        "stop_reason": "limit_reached" if len(items) >= limit else (
            "search_exhausted" if len(exhausted) == len(streams) else "page_budget_reached"
        ),
        "items": documents,
    }


def save_raw(result: dict, output_dir: Path = RAW_DIR) -> Path:
    slug = re.sub(r"[^\w-]+", "_", result["company"].casefold()).strip("_") or "company"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"hacker_news_{slug}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default="Microsoft")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    try:
        print(f"Searching Hacker News for {args.company!r}...", flush=True)
        with httpx.Client(timeout=30, headers={"User-Agent": "GradientNovaAI/0.1"}) as client:
            result = collect(args.company, args.limit, client=client)
        path = save_raw(result)
    except (httpx.HTTPError, ValueError, OSError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {result['item_count']} candidates ({result['counts']}) to {path}")
    print(f"Stopped: {result['stop_reason']}. Candidates are not verified workplace evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
