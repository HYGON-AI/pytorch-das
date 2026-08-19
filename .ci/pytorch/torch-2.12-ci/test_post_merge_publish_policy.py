import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
PIPELINE = WORKFLOWS / "_pytorch-2.12-pipeline.yml"
PR_WORKFLOW = WORKFLOWS / "pytorch-2.12-pr-test.yml"
NIGHTLY_WORKFLOW = WORKFLOWS / "pytorch-2.12-nightly.yml"
PUBLISH_WORKFLOW = WORKFLOWS / "pytorch-2.12-publish.yml"
HCU_TEST_WORKFLOW = WORKFLOWS / "_pytorch-2.12-hcu-test.yml"
TEST_RUNNER = (
    REPO_ROOT
    / ".ci"
    / "pytorch"
    / "torch-2.12-ci"
    / "run-nightly-tests.sh"
)
CONTAINER_TEST_RUNNER = (
    REPO_ROOT
    / ".ci"
    / "pytorch"
    / "torch-2.12-ci"
    / "run-nightly-test-in-container.sh"
)
TEST_SUMMARIZER = (
    REPO_ROOT
    / ".ci"
    / "pytorch"
    / "torch-2.12-ci"
    / "summarize-nightly-tests.py"
)
PREPARE_SCRIPT = (
    REPO_ROOT / ".ci" / "pytorch" / "torch-2.12-ci" / "prepare.sh"
)
FLASH_ATTN_INSTALLER = (
    REPO_ROOT
    / ".ci"
    / "pytorch"
    / "torch-2.12-ci"
    / "install-flash-attn-test-wheel.sh"
)
TARGET_BRANCH = "2.12.0-dev-opensource-V1"


class TestSplitWorkflowPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = PIPELINE.read_text(encoding="utf-8")
        cls.pr_workflow = PR_WORKFLOW.read_text(encoding="utf-8")
        cls.nightly_workflow = NIGHTLY_WORKFLOW.read_text(encoding="utf-8")
        cls.publish_workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
        cls.hcu_test_workflow = HCU_TEST_WORKFLOW.read_text(encoding="utf-8")
        cls.test_runner = TEST_RUNNER.read_text(encoding="utf-8")
        cls.container_test_runner = CONTAINER_TEST_RUNNER.read_text(
            encoding="utf-8"
        )
        cls.test_summarizer = TEST_SUMMARIZER.read_text(encoding="utf-8")
        cls.prepare_script = PREPARE_SCRIPT.read_text(encoding="utf-8")
        cls.flash_attn_installer = FLASH_ATTN_INSTALLER.read_text(
            encoding="utf-8"
        )

    def test_pr_targets_v1_and_locally_merges_exact_shas(self):
        self.assertIn(f"      - {TARGET_BRANCH}", self.pr_workflow)
        self.assertIn("suite: pr", self.pr_workflow)
        self.assertIn(
            "merge_pr: ${{ github.event_name == 'pull_request' }}",
            self.pr_workflow,
        )
        self.assertIn("github.event.pull_request.base.sha", self.pr_workflow)
        self.assertIn("github.event.pull_request.head.sha", self.pr_workflow)
        self.assertIn("Merge PR source into V1 target locally", self.pipeline)
        self.assertIn('merge --no-ff --no-edit "${PR_HEAD_SHA}"', self.pipeline)

    def test_personal_branch_push_temporarily_runs_pr_suite(self):
        self.assertIn(
            "# TEMPORARY: remove this push trigger before merging",
            self.pr_workflow,
        )
        self.assertIn("      - ljz/pytorch-2.12-ci", self.pr_workflow)
        self.assertIn("github.event_name == 'push'", self.pr_workflow)
        self.assertIn("source_ref: ${{ github.sha }}", self.pr_workflow)

    def test_pr_test_case_failures_are_report_only(self):
        self.assertIn("publish: false", self.pr_workflow)
        self.assertNotIn("Fail PR after reporting all test failures", self.pipeline)
        self.assertNotIn("fail-after-summary", self.test_runner)
        self.assertNotIn("fail-after-summary", self.test_summarizer)
        self.assertIn('test_failure_policy="report-only"', self.test_runner)
        self.assertIn('test_failure_policy = "report-only"', self.test_summarizer)

    def test_nightly_builds_v1_and_runs_full_suite(self):
        self.assertIn('cron: "0 18 * * 0,2,4"', self.nightly_workflow)
        self.assertIn("suite: nightly", self.nightly_workflow)
        self.assertIn(f"source_ref: {TARGET_BRANCH}", self.nightly_workflow)

    def test_publish_is_a_separate_target_push_workflow(self):
        self.assertIn("push:", self.publish_workflow)
        self.assertIn(f"      - {TARGET_BRANCH}", self.publish_workflow)
        self.assertIn("suite: publish", self.publish_workflow)
        self.assertIn("publish: true", self.publish_workflow)
        self.assertIn("secrets.PYPI_PASSWORD", self.publish_workflow)
        self.assertIn("inputs.publish", self.pipeline)
        self.assertIn("Upload both wheels to DevPI after merged-commit build", self.pipeline)
        self.assertIn("steps.test_summary.outcome == 'success'", self.pipeline)

    def test_test_image_is_bound_to_the_locally_compiled_sha(self):
        self.assertIn("source_sha: ${{ needs.build.outputs.source_sha }}", self.pipeline)
        self.assertIn("EXPECTED_SOURCE_SHA: ${{ inputs.source_sha }}", self.hcu_test_workflow)
        self.assertIn('image_source_sha}" != "${EXPECTED_SOURCE_SHA}', self.hcu_test_workflow)

    def test_local_docker_handoff_does_not_use_github_artifacts(self):
        combined = "\n".join(
            (
                self.pipeline,
                self.pr_workflow,
                self.nightly_workflow,
                self.publish_workflow,
            )
        )
        self.assertNotIn("actions/upload-artifact", combined)
        self.assertNotIn("actions/download-artifact", combined)

    def test_flash_attention_matches_the_torch_212_test_wheel(self):
        self.assertIn('FLASH_ATTN_TEST_VERSION: "2.8.3"', self.pipeline)
        self.assertIn('FLASH_ATTN_TORCH_TAG: "torch2120"', self.pipeline)
        self.assertIn("resolve_flash_attn_wheel.py", self.pipeline)
        self.assertIn("install-flash-attn-test-wheel.sh", self.pipeline)
        self.assertIn("--no-deps", self.flash_attn_installer)
        self.assertIn("FLASH_ATTN_EXPECTED_TORCH_TAG", self.flash_attn_installer)
        self.assertIn("flash_attn_2_cuda", self.flash_attn_installer)
        self.assertNotIn("import pathlib, torch", self.flash_attn_installer)
        self.assertIn(
            'importlib.util.find_spec("torch")', self.flash_attn_installer
        )
        self.assertIn(
            "/opt/intel/oneapi/mkl/2024.2/lib", self.flash_attn_installer
        )
        self.assertIn(
            'name == "libflash_attention.so"', self.flash_attn_installer
        )
        self.assertIn("find /opt/dtk", self.flash_attn_installer)
        self.assertIn("-name 'libomp.so'", self.flash_attn_installer)
        self.assertIn(
            "/opt/flash_attn_runtime_library_path",
            self.flash_attn_installer,
        )
        self.assertIn(
            "/opt/flash_attn_runtime_library_path",
            self.container_test_runner,
        )
        self.assertIn(
            "flash_attn_runtime_library_path", self.container_test_runner
        )
        self.assertNotIn(
            "import flash_attn_2_cuda", self.flash_attn_installer
        )
        self.assertIn(
            "import flash_attn_2_cuda", self.container_test_runner
        )
        self.assertLess(
            self.flash_attn_installer.index("export LD_LIBRARY_PATH="),
            self.flash_attn_installer.index(
                'ldd "${FLASH_ATTN_LINK_PATH}"'
            ),
        )
        self.assertLess(
            self.container_test_runner.index(
                "if not torch.cuda.is_available()"
            ),
            self.container_test_runner.index("import flash_attn_2_cuda"),
        )
        self.assertNotIn("20260107_nmz_tmp", self.prepare_script)


if __name__ == "__main__":
    unittest.main()
