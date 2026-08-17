import os
import re

import pytest


SKIP_FILE_ENV = "PYTORCH_DCU_PR_SKIP_FILE"
CLAUSE_PATTERN = re.compile(
    r"^[a-zA-Z0-9_.-]+(?:\s+and\s+[a-zA-Z0-9_.-]+)*$"
)


def load_skip_clauses(path):
    clauses = []
    with open(path, encoding="utf-8") as skip_file:
        for line_number, line in enumerate(skip_file, 1):
            clause = line.strip()
            if not clause or clause.startswith("#"):
                continue
            if not CLAUSE_PATTERN.fullmatch(clause):
                raise ValueError(
                    "invalid DCU PR skip clause at {}:{}: {}".format(
                        path, line_number, clause
                    )
                )
            clauses.append(clause)
    if not clauses:
        raise ValueError("no active DCU PR skip clauses in {}".format(path))
    return clauses


def clause_matches_nodeid(clause, nodeid):
    normalized_nodeid = nodeid.lower()
    return all(
        term.strip().lower() in normalized_nodeid
        for term in clause.split(" and ")
    )


def pytest_collection_modifyitems(items):
    skip_file = os.environ.get(SKIP_FILE_ENV)
    if not skip_file:
        return

    clauses = load_skip_clauses(skip_file)
    for item in items:
        matching_clause = next(
            (
                clause
                for clause in clauses
                if clause_matches_nodeid(clause, item.nodeid)
            ),
            None,
        )
        if matching_clause is not None:
            item.add_marker(
                pytest.mark.skip(
                    reason="DCU PR reviewed skip: {}".format(matching_clause)
                )
            )
