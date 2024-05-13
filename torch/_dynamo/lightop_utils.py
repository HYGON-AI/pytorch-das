import torch
from torch._inductor.lightop import (patch_functions,
                                     fuselayernorm_replaces_layernorm,
                                     fuse_ln_dropout_fusion)


def lightop_replace(gm: torch.fx.GraphModule):
    with patch_functions():
        gm = fuse_ln_dropout_fusion(gm)
        gm = fuselayernorm_replaces_layernorm(gm)
    return gm
