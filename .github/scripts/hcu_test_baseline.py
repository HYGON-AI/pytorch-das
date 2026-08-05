#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import re
from pathlib import Path


NODE_ID = re.compile(
    r"^(?P<file>[^:*?\[\]]+\.py)::(?P<class>[A-Za-z_][A-Za-z0-9_]*)::"
    r"(?P<test>test_[^:*?\[\]]+)$"
)
REQUIRED_FIELDS = {
    "test",
    "owner",
    "expires",
    "category",
    "reason",
    "tracking",
}


def load_and_validate(
    baseline_path: Path, repo_root: Path, as_of: dt.date | None = None
) -> list[dict[str, str]]:
    if as_of is None:
        as_of = dt.date.today()

    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("HCU test baseline must be a JSON list")

    seen: set[str] = set()
    validated: list[dict[str, str]] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"baseline entry {index} must be an object")
        missing = REQUIRED_FIELDS - entry.keys()
        extra = entry.keys() - REQUIRED_FIELDS
        if missing or extra:
            raise ValueError(
                f"baseline entry {index} has missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )

        if not all(isinstance(entry[field], str) for field in REQUIRED_FIELDS):
            raise ValueError(f"baseline entry {index} fields must all be strings")

        node_id = entry["test"]
        match = NODE_ID.fullmatch(node_id)
        if not match:
            raise ValueError(
                f"baseline entry {index} must name one exact file/class/test node: "
                f"{node_id!r}"
            )
        if node_id in seen:
            raise ValueError(f"duplicate HCU baseline node: {node_id}")
        seen.add(node_id)

        test_file = repo_root / match.group("file")
        if not test_file.is_file():
            raise ValueError(f"baseline test file does not exist: {test_file}")
        if not entry["owner"].strip():
            raise ValueError(f"baseline owner is empty for {node_id}")
        if not entry["category"].strip() or not entry["reason"].strip():
            raise ValueError(f"baseline category/reason is empty for {node_id}")
        if not entry["tracking"].startswith("https://"):
            raise ValueError(f"baseline tracking must be an HTTPS URL for {node_id}")

        try:
            expiry = dt.date.fromisoformat(entry["expires"])
        except ValueError as error:
            raise ValueError(f"invalid expiry for {node_id}: {entry['expires']}") from error
        if expiry < as_of:
            raise ValueError(
                f"expired HCU baseline node {node_id}: {expiry} is before {as_of}"
            )

        validated.append(entry)

    return validated


def to_disabled_tests(entries: list[dict[str, str]]) -> dict[str, list[object]]:
    disabled: dict[str, list[object]] = {}
    for entry in entries:
        match = NODE_ID.fullmatch(entry["test"])
        if match is None:
            raise AssertionError(f"validated node no longer matches: {entry['test']}")
        module = match.group("file")[:-3].replace("/", ".")
        key = f"{match.group('test')} ({module}.{match.group('class')})"
        disabled[key] = [entry["tracking"], ["rocm"]]
    return disabled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and emit the HCU test baseline")
    parser.add_argument("command", choices=("validate", "emit-disabled"))
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--as-of", type=dt.date.fromisoformat)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_and_validate(args.baseline, args.repo_root, args.as_of)
    if args.command == "validate":
        print(f"Validated {len(entries)} exact HCU baseline nodes")
        return

    if args.output is None:
        raise SystemExit("--output is required for emit-disabled")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(to_disabled_tests(entries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(entries)} exact disabled HCU nodes to {args.output}")


if __name__ == "__main__":
    main()
