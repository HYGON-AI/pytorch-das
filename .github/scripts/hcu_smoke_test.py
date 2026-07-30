#!/usr/bin/env python3

import os
from pathlib import Path

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


class HcuRuntimeTest(TestCase):
    def test_installed_wheel_and_hcu_runtime(self) -> None:
        source_root = Path(os.environ["HCU_SOURCE_ROOT"]).resolve()
        torch_path = Path(torch.__file__).resolve()
        try:
            torch_path.relative_to(source_root)
        except ValueError:
            pass
        else:
            self.fail(f"torch was imported from the source checkout: {torch_path}")

        expected_device_count = int(os.environ["HCU_EXPECTED_DEVICE_COUNT"])
        self.assertTrue(torch.version.hip)
        self.assertTrue(torch.cuda.is_available())
        self.assertGreaterEqual(torch.cuda.device_count(), expected_device_count)
        self.assertTrue(torch.cuda.get_device_name(0))

        print(f"torch_version={torch.__version__}")
        print(f"torch_path={torch_path}")
        print(f"hip_version={torch.version.hip}")
        print(f"hcu_device_count={torch.cuda.device_count()}")
        print(f"hcu_device_0={torch.cuda.get_device_name(0)}")


class HcuSmokeTest(TestCase):
    def test_basic_tensor_computation(self, device: str) -> None:
        lhs = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rhs = torch.arange(16, 32, dtype=torch.float32).reshape(4, 4)
        expected = torch.relu(lhs @ rhs + 1).sum()

        actual = torch.relu(lhs.to(device) @ rhs.to(device) + 1).sum()
        torch.cuda.synchronize(device)

        self.assertEqual(actual.cpu(), expected)

    def test_tiny_training_step(self, device: str) -> None:
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        inputs = torch.randn(4, 8, device=device)
        targets = torch.randn(4, 4, device=device)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        optimizer.step()
        torch.cuda.synchronize(device)

        total_update = sum(
            (parameter.detach() - original).abs().sum().item()
            for parameter, original in zip(model.parameters(), before)
        )
        self.assertGreater(total_update, 0.0)


instantiate_device_type_tests(HcuSmokeTest, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
