#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Union


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def collect_junit_stats(result_root: Path) -> Dict[str, Union[int, float]]:
    stats: Dict[str, Union[int, float]] = {
        "xml_files": 0,
        "invalid_xml_files": 0,
        "total": 0,
        "passed": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "duration_seconds": 0.0,
    }

    xml_files: Iterable[Path] = sorted(
        result_root.glob("*/test-reports/**/*.xml")
    )
    for xml_file in xml_files:
        stats["xml_files"] += 1
        try:
            root = ET.parse(str(xml_file)).getroot()
        except (ET.ParseError, OSError):
            stats["invalid_xml_files"] += 1
            continue

        for testcase in root.iter():
            if _local_name(testcase.tag) != "testcase":
                continue

            stats["total"] += 1
            try:
                stats["duration_seconds"] += float(
                    testcase.attrib.get("time", "0") or 0
                )
            except ValueError:
                pass

            child_names = {
                _local_name(child.tag) for child in list(testcase)
            }
            if "error" in child_names:
                stats["errors"] += 1
            elif "failure" in child_names:
                stats["failures"] += 1
            elif "skipped" in child_names:
                stats["skipped"] += 1
            else:
                stats["passed"] += 1

    executed = int(stats["passed"]) + int(stats["failures"]) + int(stats["errors"])
    stats["success_rate"] = (
        float(stats["passed"]) * 100.0 / executed if executed else 0.0
    )
    return stats


def junit_summary_lines(
    stats: Dict[str, Union[int, float]],
) -> Iterable[str]:
    return (
        f"total_tests={stats['total']}",
        f"passed_tests={stats['passed']}",
        f"failed_tests={stats['failures']}",
        f"error_tests={stats['errors']}",
        f"skipped_tests={stats['skipped']}",
        f"success_rate={float(stats['success_rate']):.2f}%",
        f"duration_seconds={float(stats['duration_seconds']):.2f}",
        f"junit_xml_files={stats['xml_files']}",
        f"invalid_junit_xml_files={stats['invalid_xml_files']}",
    )


def append_junit_step_summary(
    summary_path: Path,
    stats: Dict[str, Union[int, float]],
) -> None:
    with summary_path.open("a", encoding="utf-8") as summary:
        summary.write("## DCU test case summary\n\n")
        summary.write(
            "| Total | Passed | Failed | Errors | Skipped | Success rate | Duration |\n"
        )
        summary.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        summary.write(
            f"| {stats['total']} | {stats['passed']} | {stats['failures']} | "
            f"{stats['errors']} | {stats['skipped']} | "
            f"{float(stats['success_rate']):.2f}% | "
            f"{float(stats['duration_seconds']):.2f}s |\n"
        )
        summary.write(
            f"\nJUnit XML files: {stats['xml_files']}; "
            f"invalid XML files: {stats['invalid_xml_files']}.\n"
        )
