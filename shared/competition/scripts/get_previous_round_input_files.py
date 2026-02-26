#!/usr/bin/env python3
"""
Fetch the first page of presigned URLs for a competition's previous-round input files.

Uses the same authenticated client as the CLI: loads hotkey from .apex.config.json
(searched by walking up from the script directory to repo root). Requires apex link.

Usage:
  uv run --group dev python shared/competition/scripts/get_previous_round_input_files.py --competition-id 1
  python shared/competition/scripts/get_previous_round_input_files.py -c 1 -l 50 -o 0

Orchestrator URL comes from common.settings (e.g. .env: ENV, ORCHESTRATOR_HOST, etc.).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path


# Find repo root by walking up until we find .apex.config.json (same place apex CLI looks when run from repo root)
def _find_apex_config_dir() -> Path:
    path = Path(__file__).resolve().parent
    for _ in range(10):
        config_file = path / ".apex.config.json"
        if config_file.exists():
            return path
        parent = path.parent
        if parent == path:
            break
        path = parent
    return Path.cwd()  # fallback to cwd


_REPO_ROOT = _find_apex_config_dir()


def _main():
    parser = argparse.ArgumentParser(
        description="Get first page of presigned URLs for a competition's previous-round input files.",
    )
    parser.add_argument("--competition-id", "-c", type=int, required=True, help="Competition ID")
    parser.add_argument("--limit", "-l", type=int, default=50, help="Page size (default: 50)")
    parser.add_argument("--offset", "-o", type=int, default=0, help="Page offset (default: 0)")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the JSON response, no summary",
    )
    args = parser.parse_args()

    # Load CLI config and client (same as CLI commands)
    try:
        from cli.utils.config import Config
        from cli.utils.client import Client
    except ImportError as e:
        print(
            "CLI packages not found. Run from repo root with: uv run --group dev python shared/competition/scripts/get_previous_round_input_files.py ...",
            file=sys.stderr,
        )
        print(f"ImportError: {e}", file=sys.stderr)
        sys.exit(1)

    config_file = _REPO_ROOT / ".apex.config.json"
    config = Config.load_config(config_file_path=config_file)
    if not config.hotkey_file_path:
        print(
            "No hotkey linked. Run `apex link` to link your wallet and hotkey, then try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    async def run():
        async with Client(hotkey_file_path=config.hotkey_file_path, timeout=config.timeout) as client:
            params = {
                "competition_id": args.competition_id,
                "limit": args.limit,
                "offset": args.offset,
            }
            response = await client._make_request(
                method="GET",
                path=f"/miner/competitions/{args.competition_id}/previous-round-input-files",
                params=params,
            )
            return response.json()

    try:
        data = asyncio.run(run())
    except Exception as e:
        err_cls = type(e).__name__
        if hasattr(e, "response") and e.response is not None:
            print(f"{err_cls}: HTTP {e.response.status_code}", file=sys.stderr)
            try:
                body = e.response.json()
                print(json.dumps(body, indent=2), file=sys.stderr)
            except Exception:
                print(e.response.text or "", file=sys.stderr)
        else:
            print(f"{err_cls}: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(data, indent=2))

    if not args.json_only and data.get("files"):
        print("\n# Download URLs (first page)", file=sys.stderr)
        for f in data["files"]:
            print(f.get("download_url", ""), file=sys.stderr)
        print(
            f"\nTotal: {data.get('total_count', 0)} files (showing {len(data['files'])}). Use --offset and --limit for more pages.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    _main()
