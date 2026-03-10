# Owner(s): ["module: inductor"]
import logging
import unittest

import torch
import torch._logging
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CUDA, HAS_GPU


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x=None):
        y = x.reshape(-1,2,16,4)
        z = y.reshape(-1,128)
        o = z + 2

        return o




class ReshapePssTest(TestCase):
    @unittest.skipIf(not HAS_GPU, "Triton is not available")
    def test_mlp(self):
        torch._logging.set_logs(
            dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG
        )
        mlp = MLP().to(GPU_TYPE)
        c_mlp = torch.compile(MLP().to(GPU_TYPE))
        x = torch.rand(8,64,2).to(GPU_TYPE)

        ref_out = mlp(x)
        act_out = c_mlp(x)
        
        self.assertEqual(ref_out, act_out)

    @unittest.skipIf(not HAS_GPU, "Triton is not available")
    def test_mlp_notcontiguous(self):
        torch._logging.set_logs(
            dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG
        )
        mlp = MLP().to(GPU_TYPE)
        c_mlp = torch.compile(MLP().to(GPU_TYPE))
        x = torch.rand(4,16,2,8).permute(0,2,1,3).to(GPU_TYPE)
        ref_out = mlp(x)
        act_out = c_mlp(x)
        
        self.assertEqual(ref_out, act_out)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if IS_LINUX and HAS_GPU:
        if (not HAS_CUDA) or torch.cuda.get_device_properties(0).major <= 5:
            run_tests()

