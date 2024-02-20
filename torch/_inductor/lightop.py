import torch
import logging
import torch.fx as fx
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
from torch.overrides import TorchFunctionMode
from torch._inductor.fx_utils import matches_module_function_pattern
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.nn import functional as F
log = logging.getLogger(__name__)

class AutogradMonkeypatch(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if not kwargs:
            kwargs = {}
        return func(*args, **kwargs)


patch_functions = AutogradMonkeypatch

def fetch_attr(target: str, mod):
    target_atoms = target.split(".")
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr

# Works for length 1 patterns with 1 modules
def matches_one_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[fx.Node] = (node,)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True

#heck the pattern: (F.function/torch.Tensor.method) matched.
# Works for length 1 patterns with 1 function/method.
def matches_one_function_pattern(pattern, node):
    if len(node.args) == 0:
        return False
    if not isinstance(node, torch.fx.Node):
        return False
    # the node is call_function or call_method
    if node.op != "call_function" and node.op != "call_method":
        return False
    if node.target != pattern[0]:
        return False
    return True

def fuselayernorm_replaces_layernorm(gm: torch.fx.GraphModule):
    modules_patterns = [(torch.nn.LayerNorm,),]
    function_patterns = [(F.layer_norm,),]
    modules = dict(gm.named_modules())
    for pattern in modules_patterns:
        for node in gm.graph.nodes:
            if matches_one_module_pattern(pattern, node, modules):
                layer_norm = modules[node.target]
                new_node = torch.nn.FuseLayerNorm(layer_norm.normalized_shape)
                new_node.weight = torch.nn.Parameter(layer_norm.weight, layer_norm.weight.requires_grad)
                new_node.bias = torch.nn.Parameter(layer_norm.bias, layer_norm.bias.requires_grad)
                new_node.eps = layer_norm.eps
                new_node.elementwise_affine = layer_norm.elementwise_affine
                replace_node_module(node, modules, new_node)
    gm.graph.lint()
    for pattern in function_patterns:
        for node in gm.graph.nodes:
            if matches_one_function_pattern(pattern, node):
                if len(node.args) != 5:
                    continue
                normalized_shape = node.args[1]
                eps = node.args[4]
                weight = fetch_attr(node.args[2].target, gm)
                bias = fetch_attr(node.args[3].target, gm)
                new_node = F.fuse_layer_norm(normalized_shape)
                new_node.weight = torch.nn.Parameter(weight. weight.requires_grad)
                new_node.bias = torch.nn.Parameter(bias, bias.requires_grad)
                new_node.eps = eps
                replace_node_module(node, modules, new_node)
    gm.graph.lint()
    gm.recompile()

    return gm
