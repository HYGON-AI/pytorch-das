#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize local PyTorch DCU nightly shard results."
    )
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--expected-shards", required=True, type=int)
    parser.add_argument("--step-summary", type=Path)
    return parser.parse_args()


def entry_name(status: dict[str, Any]) -> str:
    return (
        f"{status.get('config')}:{status.get('shard')}/"
        f"{status.get('num_shards')}:rc={status.get('exit_code')}"
    )


def main() -> int:
    args = parse_args()
    status_files = sorted(args.result_root.glob("*/status.json"))

    passed: list[str] = []
    test_failed: list[str] = []
    infrastructure_failed: list[str] = []
    unknown: list[str] = []
    matrix_entries: set[tuple[Any, Any, Any]] = set()

    for status_file in status_files:
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception as error:
            unknown.append(f"{status_file.parent.name}: invalid status ({error})")
            continue

        entry = entry_name(status)
        matrix_entry = (
            status.get("config"),
            status.get("shard"),
            status.get("num_shards"),
        )
        if matrix_entry in matrix_entries:
            unknown.append(f"{entry}: duplicate matrix entry")
            continue
        matrix_entries.add(matrix_entry)

        failure_kind = status.get("failure_kind", "unknown")
        conclusion = status.get("conclusion")
        exit_code = status.get("exit_code")
        tests_started = status.get("tests_started") == 1
        tests_completed = status.get("tests_completed") == 1

        if (
            failure_kind == "none"
            and conclusion == "passed"
            and exit_code == 0
            and tests_started
            and tests_completed
        ):
            passed.append(entry)
        elif (
            failure_kind == "test"
            and conclusion == "failed"
            and isinstance(exit_code, int)
            and exit_code != 0
            and tests_started
            and tests_completed
        ):
            test_failed.append(entry)
        elif failure_kind == "infrastructure":
            infrastructure_failed.append(entry)
        else:
            unknown.append(entry)

    missing_status_shards = max(args.expected_shards - len(status_files), 0)
    unexpected_status_files = max(len(status_files) - args.expected_shards, 0)
    infrastructure_count = (
        len(infrastructure_failed)
        + len(unknown)
        + missing_status_shards
        + unexpected_status_files
    )
    lines = [
        "test_failure_policy=report-only",
        f"expected_shards={args.expected_shards}",
        f"observed_status_files={len(status_files)}",
        f"completed_test_shards={len(passed) + len(test_failed)}",
        f"missing_status_shards={missing_status_shards}",
        f"unexpected_status_files={unexpected_status_files}",
        f"passed_shards={len(passed)}",
        f"failed_test_shards={len(test_failed)}",
        f"failed_infrastructure_shards={infrastructure_count}",
    ]
    if test_failed:
        lines.append("failed_test_entries:")
        lines.extend(f"  {entry}" for entry in test_failed)
    if infrastructure_failed or unknown:
        lines.append("failed_infrastructure_entries:")
        lines.extend(f"  {entry}" for entry in infrastructure_failed + unknown)

    args.result_root.mkdir(parents=True, exist_ok=True)
    (args.result_root / "nightly-summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))

    if args.step_summary:
        with args.step_summary.open("a", encoding="utf-8") as summary:
            summary.write("## DCU test summary\n\n")
            summary.write(
                "| Policy | Expected | Completed | Passed | Test failures | "
                "Infrastructure failures |\n"
            )
            summary.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
            summary.write(
                f"| Report only | {args.expected_shards} | {len(status_files)} | "
                f"{len(passed)} | {len(test_failed)} | {infrastructure_count} |\n"
            )

    for entry in test_failed:
        print(f"::warning title=DCU nightly test failure::{entry}")
    for entry in infrastructure_failed + unknown:
        print(f"::error title=DCU test infrastructure failure::{entry}")

    if len(status_files) != args.expected_shards:
        print(
            "::error title=Incomplete DCU test matrix::"
            f"expected {args.expected_shards} status files, "
            f"found {len(status_files)}"
        )
        return 2
    if infrastructure_failed or unknown:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
