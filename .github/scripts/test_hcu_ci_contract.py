#!/usr/bin/env python3

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SETUP_HCU = REPO_ROOT / ".github" / "actions" / "setup-hcu" / "action.yml"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class HcuCiContractTest(unittest.TestCase):
    def test_required_hcu_ci_files_exist(self) -> None:
        for path in (
            WORKFLOWS / "_hcu-preflight.yml",
            WORKFLOWS / "_hcu-build.yml",
            WORKFLOWS / "_hcu-test.yml",
            WORKFLOWS / "hcu.yml",
            SETUP_HCU,
        ):
            self.assertTrue(path.is_file(), f"missing required HCU CI file: {path}")

    def test_pr_hcu_build_is_explicitly_enabled_and_preflighted(self) -> None:
        pull_workflow = read(WORKFLOWS / "pull.yml")

        self.assertIn("hcu-preflight:", pull_workflow)
        self.assertIn("linux-jammy-rocm-hcu-py3_10-build:", pull_workflow)
        self.assertIn("vars.HCU_CI_ENABLED == 'true'", pull_workflow)
        self.assertIn("uses: ./.github/workflows/_hcu-preflight.yml", pull_workflow)
        self.assertIn("needs: hcu-preflight", pull_workflow)
        self.assertIn("uses: ./.github/workflows/_hcu-build.yml", pull_workflow)

    def test_hcu_backend_workflow_builds_before_testing(self) -> None:
        workflow = read(WORKFLOWS / "hcu.yml")

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("schedule:", workflow)
        self.assertIn("ciflow/hcu/*", workflow)
        self.assertIn("hcu-build:", workflow)
        self.assertIn("hcu-test:", workflow)
        self.assertIn("needs: hcu-build", workflow)
        self.assertIn("uses: ./.github/workflows/_hcu-test.yml", workflow)

    def test_hcu_build_uses_fresh_artifact_and_hcu_test_validates_it(self) -> None:
        build_workflow = read(WORKFLOWS / "_hcu-build.yml")
        test_workflow = read(WORKFLOWS / "_hcu-test.yml")

        self.assertIn("HCU_REUSE_OLD_WHL: \"0\"", build_workflow)
        self.assertIn("hcu-build-manifest.json", build_workflow)
        self.assertIn("uses: ./.github/actions/upload-build-artifacts", build_workflow)
        self.assertIn("use-gha: true", build_workflow)
        self.assertIn("SOURCE_SHA: ${{ github.sha }}", build_workflow)
        self.assertIn("hcu_ci_manifest.py validate", test_workflow)
        self.assertIn("uses: actions/download-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", test_workflow)
        self.assertIn("SOURCE_SHA: ${{ github.sha }}", test_workflow)
        self.assertIn("torch.cuda.is_available()", test_workflow)
        self.assertNotIn('<<"PY"', test_workflow)

    def test_preflight_rejects_forked_pull_requests(self) -> None:
        workflow = read(WORKFLOWS / "_hcu-preflight.yml")

        self.assertIn("github.event.pull_request.head.repo.full_name", workflow)
        self.assertIn("github.repository", workflow)
        self.assertIn("HCU_BUILD_RUNNER", workflow)
        self.assertIn("HCU_TEST_RUNNER", workflow)
        self.assertIn("HCU_CI_IMAGE", workflow)
        self.assertIn("HCU_EXPECTED_DEVICE_COUNT", workflow)

    def test_shared_runner_setup_is_scoped_and_hcu_specific(self) -> None:
        setup = read(SETUP_HCU)

        self.assertIn("/opt/dtk", setup)
        self.assertIn("rocminfo", setup)
        self.assertIn("/opt/hyhal/bin/hy-smi", setup)
        self.assertIn("/opt/hyhal/bin/hy-smi | awk", setup)
        self.assertIn("--volume /opt/hyhal:/opt/hyhal:ro", setup)
        self.assertNotIn("rocminfo | awk", setup)
        self.assertIn("org.pytorch.hcu.run", setup)
        self.assertIn("docker rm -f", setup)
        self.assertNotIn("docker container prune", setup)
        self.assertNotIn("docker stop $(docker ps -q)", setup)
        self.assertNotIn("killall runsvc.sh", setup)

    def test_manifest_helper_runs_on_the_nmz4_host_python(self) -> None:
        helper = read(REPO_ROOT / ".github" / "scripts" / "hcu_ci_manifest.py")

        self.assertNotIn("add_subparsers(dest=\"command\", required=True)", helper)
        self.assertIn("if not args.command:", helper)

    def test_hcu_ci_never_uses_pull_request_target(self) -> None:
        for path in (
            WORKFLOWS / "pull.yml",
            WORKFLOWS / "hcu.yml",
            WORKFLOWS / "_hcu-preflight.yml",
            WORKFLOWS / "_hcu-build.yml",
            WORKFLOWS / "_hcu-test.yml",
            SETUP_HCU,
        ):
            self.assertNotIn("pull_request_target", read(path), path.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
