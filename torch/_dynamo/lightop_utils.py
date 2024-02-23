import torch
from torch._inductor.lightop import (patch_functions,
                                     fuselayernorm_replaces_layernorm)


def lightop_replace(gm: torch.fx.GraphModule):
    with patch_functions():
        gm = fuselayernorm_replaces_layernorm(gm)

    return gm
