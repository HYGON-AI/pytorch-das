#!/usr/bin/env python3

import argparse
from pathlib import Path

from junit_test_stats import (
    append_junit_step_summary,
    collect_junit_stats,
    junit_summary_lines,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize PyTorch JUnit XML test cases."
    )
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--step-summary", type=Path)
    args = parser.parse_args()

    stats = collect_junit_stats(args.result_root)
    print("\n".join(junit_summary_lines(stats)))
    if args.step_summary:
        append_junit_step_summary(args.step_summary, stats)

    if stats["invalid_xml_files"]:
        print(
            "::error title=Invalid JUnit XML::"
            f"{stats['invalid_xml_files']} report file(s) could not be parsed"
        )
    if not stats["xml_files"]:
        print(
            "::error title=Missing JUnit XML::"
            "No test-case XML reports were found"
        )
    if stats["invalid_xml_files"] or not stats["xml_files"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
