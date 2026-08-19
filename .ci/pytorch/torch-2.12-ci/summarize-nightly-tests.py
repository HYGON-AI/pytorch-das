#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize local PyTorch HCU nightly shard results."
    )
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--expected-shards", required=True, type=int)
    parser.add_argument("--expected-config")
    parser.add_argument("--step-summary", type=Path)
    parser.add_argument("--github-output", type=Path)
    return parser.parse_args()


def entry_name(status: Dict[str, Any]) -> str:
    return (
        f"{status.get('config')}:{status.get('shard')}/"
        f"{status.get('num_shards')}:rc={status.get('exit_code')}"
    )


def main() -> int:
    args = parse_args()
    status_files = sorted(args.result_root.glob("*/status.json"))

    passed: List[str] = []
    test_failed: List[str] = []
    infrastructure_failed: List[str] = []
    unknown: List[str] = []
    matrix_entries: Set[Tuple[Any, Any, Any]] = set()

    for status_file in status_files:
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception as error:
            unknown.append(f"{status_file.parent.name}: invalid status ({error})")
            continue

        entry = entry_name(status)
        config = status.get("config")
        shard = status.get("shard")
        num_shards = status.get("num_shards")
        if (
            not isinstance(config, str)
            or not isinstance(shard, int)
            or not isinstance(num_shards, int)
        ):
            unknown.append(f"{entry}: invalid matrix entry")
            continue

        matrix_entry = (config, shard, num_shards)
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

    if args.expected_config:
        expected_matrix_entries = {
            (args.expected_config, shard, args.expected_shards)
            for shard in range(1, args.expected_shards + 1)
        }
        unexpected_matrix_entries = matrix_entries - expected_matrix_entries
        for config, shard, num_shards in sorted(unexpected_matrix_entries):
            unknown.append(
                f"{config}:{shard}/{num_shards}: unexpected matrix entry"
            )

    missing_status_shards = max(args.expected_shards - len(status_files), 0)
    unexpected_status_files = max(len(status_files) - args.expected_shards, 0)
    infrastructure_count = (
        len(infrastructure_failed)
        + len(unknown)
        + missing_status_shards
        + unexpected_status_files
    )
    all_tests_passed = (
        len(status_files) == args.expected_shards
        and len(passed) == args.expected_shards
        and not test_failed
        and infrastructure_count == 0
    )
    test_failure_policy = "report-only"
    lines = [
        f"test_failure_policy={test_failure_policy}",
        f"expected_shards={args.expected_shards}",
        f"observed_status_files={len(status_files)}",
        f"completed_test_shards={len(passed) + len(test_failed)}",
        f"missing_status_shards={missing_status_shards}",
        f"unexpected_status_files={unexpected_status_files}",
        f"passed_shards={len(passed)}",
        f"failed_test_shards={len(test_failed)}",
        f"failed_infrastructure_shards={infrastructure_count}",
        f"all_tests_passed={str(all_tests_passed).lower()}",
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

    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(
                f"all_tests_passed={str(all_tests_passed).lower()}\n"
            )
            output.write(f"passed_shards={len(passed)}\n")
            output.write(f"failed_test_shards={len(test_failed)}\n")
            output.write(
                f"failed_infrastructure_shards={infrastructure_count}\n"
            )

    if args.step_summary:
        with args.step_summary.open("a", encoding="utf-8") as summary:
            summary.write("## HCU test summary\n\n")
            summary.write(
                "| Policy | Expected | Completed | Passed | Test failures | "
                "Infrastructure failures |\n"
            )
            summary.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
            summary.write(
                f"| {test_failure_policy} | {args.expected_shards} | {len(status_files)} | "
                f"{len(passed)} | {len(test_failed)} | {infrastructure_count} |\n"
            )

    for entry in test_failed:
        print(f"::warning title=HCU test failure::{entry}")
    for entry in infrastructure_failed + unknown:
        print(f"::error title=HCU test infrastructure failure::{entry}")

    if len(status_files) != args.expected_shards:
        print(
            "::error title=Incomplete HCU test matrix::"
            f"expected {args.expected_shards} status files, "
            f"found {len(status_files)}"
        )
        return 2
    if infrastructure_failed or unknown:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
