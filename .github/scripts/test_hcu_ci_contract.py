#!/usr/bin/env python3

import re
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SETUP_HCU = REPO_ROOT / ".github" / "actions" / "setup-hcu" / "action.yml"
SMOKE_TEST = REPO_ROOT / ".github" / "scripts" / "hcu_smoke_test.py"
MANIFEST_HELPER = REPO_ROOT / ".github" / "scripts" / "hcu_ci_manifest.py"
MANIFEST_TEST = REPO_ROOT / ".github" / "scripts" / "test_hcu_ci_manifest.py"
ATEN_CMAKE = REPO_ROOT / "aten" / "src" / "ATen" / "CMakeLists.txt"
HCU_STATIC = WORKFLOWS / "hcu-static.yml"
LABELER = REPO_ROOT / ".github" / "labeler.yml"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class HcuCiContractTest(unittest.TestCase):
    def test_required_hcu_ci_files_exist(self) -> None:
        for path in (
            WORKFLOWS / "_hcu-preflight.yml",
            WORKFLOWS / "_hcu-build.yml",
            WORKFLOWS / "_hcu-test.yml",
            WORKFLOWS / "hcu.yml",
            HCU_STATIC,
            SETUP_HCU,
            SMOKE_TEST,
            MANIFEST_HELPER,
            MANIFEST_TEST,
            LABELER,
        ):
            self.assertTrue(path.is_file(), f"missing required HCU CI file: {path}")

    def test_hcu_yaml_files_parse(self) -> None:
        for path in (
            WORKFLOWS / "_hcu-preflight.yml",
            WORKFLOWS / "_hcu-build.yml",
            WORKFLOWS / "_hcu-test.yml",
            WORKFLOWS / "hcu.yml",
            HCU_STATIC,
            SETUP_HCU,
            LABELER,
        ):
            self.assertIsInstance(yaml.safe_load(read(path)), dict, path.name)

    def test_hygon_static_workflow_uses_the_configured_control_runner(self) -> None:
        workflow = read(HCU_STATIC)

        self.assertIn("pull_request:", workflow)
        self.assertIn("push:", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn(
            'runs-on: [self-hosted, linux, x64, "${{ vars.HCU_CONTROL_RUNNER }}"]',
            workflow,
        )
        self.assertNotIn("runs-on: ubuntu-latest", workflow)
        self.assertIn(
            "github.event.pull_request.head.repo.full_name == github.repository",
            workflow,
        )
        self.assertIn("cancel-in-progress: true", workflow)
        self.assertIn("python3 .github/scripts/test_hcu_ci_contract.py", workflow)
        self.assertIn("python3 .github/scripts/test_hcu_ci_manifest.py", workflow)
        self.assertIn("spin quicklint", workflow)
        self.assertIn("ACTIONLINT,CMAKE,RUFF,SHELLCHECK", workflow)
        self.assertIn("packaging==24.0", workflow)
        self.assertIn("spin==0.17", workflow)
        self.assertIn("typing-extensions==4.12.2", workflow)
        self.assertIn("uv==0.12.0", workflow)
        self.assertNotIn("HCU_BUILD_RUNNER", workflow)
        self.assertNotIn("HCU_TEST_RUNNER", workflow)

    def test_hcu_label_covers_static_workflow_and_scripts(self) -> None:
        labeler = read(LABELER)

        self.assertIn('"ciflow/hcu":', labeler)
        self.assertIn(".github/workflows/hcu*.yml", labeler)
        self.assertIn(".github/workflows/_hcu-*.yml", labeler)
        self.assertIn(".github/scripts/hcu_*.py", labeler)
        self.assertIn(".github/scripts/test_hcu_ci_*.py", labeler)

    def test_external_actions_are_pinned_to_commits(self) -> None:
        for path in (
            WORKFLOWS / "_hcu-build.yml",
            WORKFLOWS / "_hcu-test.yml",
            WORKFLOWS / "hcu.yml",
            HCU_STATIC,
        ):
            for action in re.findall(r"^\s*uses:\s*([^\s#]+)", read(path), re.MULTILINE):
                if action.startswith("./"):
                    continue
                self.assertRegex(action, r"^[^@]+@[0-9a-f]{40}$", path.name)

    def test_pr_hcu_build_is_explicitly_enabled_and_preflighted(self) -> None:
        pull_workflow = read(WORKFLOWS / "pull.yml")

        self.assertIn("hcu-preflight:", pull_workflow)
        self.assertIn("linux-jammy-rocm-hcu-py3_10-build:", pull_workflow)
        self.assertIn("vars.HCU_CI_ENABLED == 'true'", pull_workflow)
        self.assertIn("control-runner: ${{ vars.HCU_CONTROL_RUNNER }}", pull_workflow)
        self.assertIn(
            "github.event.pull_request.head.repo.full_name == github.repository",
            pull_workflow,
        )
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
        self.assertEqual(workflow.count('"config":'), 1)
        self.assertIn('"config":"smoke"', workflow)

    def test_hcu_build_uses_fresh_artifact_and_hcu_test_validates_it(self) -> None:
        build_workflow = read(WORKFLOWS / "_hcu-build.yml")
        test_workflow = read(WORKFLOWS / "_hcu-test.yml")
        backend_workflow = read(WORKFLOWS / "hcu.yml")
        pull_workflow = read(WORKFLOWS / "pull.yml")

        self.assertIn('HCU_REUSE_OLD_WHL: "0"', build_workflow)
        self.assertIn('CI: "1"', build_workflow)
        self.assertIn('USE_KINETO: "0"', build_workflow)
        self.assertIn('USE_FLASH_ATTENTION: "0"', build_workflow)
        self.assertIn('USE_MEM_EFF_ATTENTION: "0"', build_workflow)
        self.assertIn('USE_ROCM_CK_GEMM: "0"', build_workflow)
        self.assertIn("PYTORCH_ROCM_ARCH: ${{ inputs.rocm-arch }}", build_workflow)
        self.assertIn("MAX_JOBS: ${{ inputs.max-jobs }}", build_workflow)
        self.assertIn("--env PYTORCH_ROCM_ARCH", build_workflow)
        self.assertIn("--env MAX_JOBS", build_workflow)
        self.assertIn("--env USE_FLASH_ATTENTION", build_workflow)
        self.assertIn("--env USE_KINETO", build_workflow)
        self.assertIn("--env USE_MEM_EFF_ATTENTION", build_workflow)
        self.assertIn("--env USE_ROCM_CK_GEMM", build_workflow)
        self.assertIn('HCU_SETUPTOOLS_VERSION: "79.0.1"', build_workflow)
        self.assertIn("--env HCU_SETUPTOOLS_VERSION", build_workflow)
        self.assertIn('"setuptools==${HCU_SETUPTOOLS_VERSION}"', build_workflow)
        self.assertIn("test -x /opt/dtk/bin/hipcc", build_workflow)
        self.assertIn(
            "ln -s librt.so.1 /usr/lib/x86_64-linux-gnu/librt.so",
            build_workflow,
        )
        self.assertIn(
            "git config --global --add safe.directory /var/lib/jenkins/workspace",
            build_workflow,
        )
        self.assertIn("hcu-build-manifest.json", build_workflow)
        self.assertIn("--volume /opt/hyhal:/opt/hyhal:ro", build_workflow)
        self.assertIn(
            "/home/github/lyt/HYGON-AI_repositories/pytorch-das-ci-artifacts",
            build_workflow,
        )
        self.assertIn("realpath -m", build_workflow)
        self.assertIn("artifacts.zip.sha256", build_workflow)
        self.assertIn('touch "${artifact_path}/INCOMPLETE"', build_workflow)
        self.assertIn('touch "${ARTIFACT_PATH}/READY"', build_workflow)
        self.assertIn('rm -- "${ARTIFACT_PATH}/INCOMPLETE"', build_workflow)
        self.assertIn("SOURCE_SHA: ${{ github.sha }}", build_workflow)
        self.assertIn("artifact-path: ${{ needs.hcu-build.outputs.artifact-path }}", backend_workflow)
        self.assertIn("build-runner-name: ${{ needs.hcu-build.outputs.runner-name }}", backend_workflow)
        self.assertIn(
            "artifact-root: /home/github/lyt/HYGON-AI_repositories/pytorch-das-ci-artifacts",
            backend_workflow,
        )
        self.assertIn(
            "artifact-root: /home/github/lyt/HYGON-AI_repositories/pytorch-das-ci-artifacts",
            pull_workflow,
        )
        self.assertIn("hcu_ci_manifest.py validate", test_workflow)
        self.assertIn("sha256sum --check artifacts.zip.sha256", test_workflow)
        self.assertIn('"${RUNNER_NAME}" != "${BUILD_RUNNER_NAME}"', test_workflow)
        self.assertIn('test -f "${artifact_path}/READY"', test_workflow)
        self.assertIn("SOURCE_SHA: ${{ github.sha }}", test_workflow)
        self.assertIn("hcu_smoke_test.py", test_workflow)
        self.assertNotIn("actions/upload-artifact@", build_workflow)
        self.assertNotIn("actions/download-artifact@", test_workflow)
        self.assertNotIn("actions/upload-artifact@", test_workflow)
        self.assertNotIn("uses: ./.github/actions/upload-build-artifacts", build_workflow)
        self.assertNotIn('<<"PY"', test_workflow)

    def test_hcu_image_is_pinned_to_an_immutable_digest(self) -> None:
        preflight = read(WORKFLOWS / "_hcu-preflight.yml")

        self.assertIn("HCU_CI_IMAGE", preflight)
        self.assertIn("@sha256:[0-9a-f]{64}", preflight)

    def test_disabled_flash_attention_excludes_hip_cutlass_sources(self) -> None:
        aten_cmake = read(ATEN_CMAKE)
        cutlass_glob = aten_cmake.index(
            'file(GLOB flash_attention_hip_cutlass_hip '
            '"native/transformers/hip/flash_attn/cutlass/*.hip")'
        )
        flash_option = aten_cmake.index("if(USE_FLASH_ATTENTION)", cutlass_glob)
        cutlass_append = aten_cmake.index(
            "list(APPEND native_transformers_hip_hip "
            "${flash_attention_hip_cutlass_hip})",
            cutlass_glob,
        )

        self.assertNotIn(
            "${flash_attention_hip_cutlass_hip}",
            aten_cmake[cutlass_glob + 1 : flash_option],
        )
        self.assertGreater(cutlass_append, flash_option)

    def test_smoke_installs_and_tests_outside_the_source_checkout(self) -> None:
        test_workflow = read(WORKFLOWS / "_hcu-test.yml")
        smoke_test = read(SMOKE_TEST)

        self.assertIn("python3 -m pip install", test_workflow)
        self.assertIn("expecttest==0.3.0", test_workflow)
        self.assertIn("opt-einsum==3.4.0", test_workflow)
        self.assertIn("--force-reinstall --no-deps", test_workflow)
        self.assertIn('install_dir="/tmp/pytorch-hcu-install-', test_workflow)
        self.assertIn('smoke_dir="/tmp/pytorch-hcu-smoke-', test_workflow)
        self.assertIn("unset PYTHONPATH", test_workflow)
        self.assertIn("HCU_SOURCE_ROOT", test_workflow)
        self.assertIn("torch_path.relative_to(source_root)", smoke_test)
        self.assertIn("torch.version.hip", smoke_test)
        self.assertIn("torch.cuda.is_available()", smoke_test)
        self.assertIn("torch.cuda.device_count()", smoke_test)
        self.assertIn("torch.cuda.get_device_name(0)", smoke_test)
        self.assertIn("self.assertEqual(actual.cpu(), expected)", smoke_test)
        self.assertIn("loss.backward()", smoke_test)
        self.assertIn(
            'instantiate_device_type_tests(HcuSmokeTest, globals(), only_for="cuda")',
            smoke_test,
        )

    def test_docker_exec_output_is_persisted_on_runner(self) -> None:
        build_workflow = read(WORKFLOWS / "_hcu-build.yml")
        test_workflow = read(WORKFLOWS / "_hcu-test.yml")

        self.assertIn("tee hcu-build.log", build_workflow)
        self.assertIn("hcu-build.log.gz", build_workflow)
        self.assertIn('log_dir="${ARTIFACT_PATH}/logs"', build_workflow)
        self.assertIn("tee hcu-test.log", test_workflow)
        self.assertIn("hcu-test.log.gz", test_workflow)
        self.assertIn('log_dir="${artifact_path}/tests/', test_workflow)

    def test_preflight_rejects_forked_pull_requests(self) -> None:
        workflow = read(WORKFLOWS / "_hcu-preflight.yml")

        self.assertIn(
            'runs-on: [self-hosted, linux, x64, "${{ inputs.control-runner }}"]',
            workflow,
        )
        self.assertIn("HCU_CONTROL_RUNNER", workflow)
        self.assertIn("github.event.pull_request.head.repo.full_name", workflow)
        self.assertIn("github.repository", workflow)
        self.assertIn("HCU_BUILD_RUNNER", workflow)
        self.assertIn("HCU_TEST_RUNNER", workflow)
        self.assertIn("HCU_CI_IMAGE", workflow)
        self.assertIn("HCU_EXPECTED_DEVICE_COUNT", workflow)

    def test_shared_runner_setup_is_scoped_and_hcu_specific(self) -> None:
        setup = read(SETUP_HCU)

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
        self.assertNotIn("/opt/dtk", setup)

    def test_missing_sccache_is_ignored_only_for_hcu_builds(self) -> None:
        common_utils = read(REPO_ROOT / ".ci" / "pytorch" / "common_utils.sh")

        self.assertIn('"${BUILD_ENVIRONMENT:-}" == *hcu*', common_utils)
        self.assertIn("skipping HCU cache statistics", common_utils)
        self.assertIn("sccache is not installed", common_utils)

    def test_manifest_helper_supports_the_legacy_host_python(self) -> None:
        helper = read(REPO_ROOT / ".github" / "scripts" / "hcu_ci_manifest.py")

        self.assertNotIn('add_subparsers(dest="command", required=True)', helper)
        self.assertIn("if not args.command:", helper)

    def test_hcu_ci_never_uses_pull_request_target(self) -> None:
        for path in (
            WORKFLOWS / "pull.yml",
            WORKFLOWS / "hcu.yml",
            HCU_STATIC,
            WORKFLOWS / "_hcu-preflight.yml",
            WORKFLOWS / "_hcu-build.yml",
            WORKFLOWS / "_hcu-test.yml",
            SETUP_HCU,
        ):
            self.assertNotIn("pull_request_target", read(path), path.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
