import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github/workflows/pytorch-2.12-build.yml"
TARGET_BRANCH = "2.12.0-dev-opensource-V1"


class TestPostMergePublishPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_target_branch_push_is_enabled(self):
        self.assertIn(f"      - {TARGET_BRANCH}", self.workflow)

    def test_pr_never_uploads_wheels(self):
        self.assertNotIn(
            "needs.preflight.outputs.test_suite == 'pr' && "
            "steps.test_summary.outputs.all_tests_passed == 'true'",
            self.workflow,
        )

    def test_only_successful_target_push_uploads_wheels(self):
        self.assertIn(
            "github.event_name == 'push' && "
            f"github.ref == 'refs/heads/{TARGET_BRANCH}' && "
            "steps.test_summary.outputs.all_tests_passed == 'true'",
            self.workflow,
        )

    def test_failed_post_merge_smoke_fails_workflow(self):
        self.assertIn("Fail post-merge publish after smoke failures", self.workflow)


if __name__ == "__main__":
    unittest.main()
