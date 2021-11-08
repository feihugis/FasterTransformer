import torch
from torch import Tensor

import time

from graph_editor import graph_pattern
from typing import List

torch.classes.load_library("/datadrive/fhu/github/FasterTransformer/build_110/lib/libpyt_fastertransformer.so")

torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)
torch._C._jit_set_bailout_depth(1)


def _add_layer_norm(
  input: Tensor, residual: Tensor, bias: Tensor, normalized_shape: List[int],
  normalized_weight: Tensor, normalized_bias: Tensor, normalized_eps: float):
  x = input + bias
  x = x + residual
  return torch.nn.functional.layer_norm(x, normalized_shape, normalized_weight, normalized_bias, normalized_eps)

def _fused_add_layer_norm(
  input: Tensor, residual: Tensor, bias: Tensor, normalized_shape: List[int],
  normalized_weight: Tensor, normalized_bias: Tensor, normalized_eps: float):
  return torch.torch.ops.fastseq.fused_add_layer_norm(input, residual, normalized_weight, normalized_bias, bias)

def add_layer_norm_pattern_0() -> str:  
  return graph_pattern(_add_layer_norm)()


def fused_add_layer_norm() -> str:
  return graph_pattern(_fused_add_layer_norm)()

print(add_layer_norm_pattern_0())
print(fused_add_layer_norm())

base = """
graph(%input.1 : Tensor,
      %residual.1 : Tensor,
      %bias.1 : Tensor,
      %normalized_shape.1 : int[],
      %normalized_weight.1 : Tensor,
      %normalized_bias.1 : Tensor,
      %normalized_eps.1 : Tensor):
  %21 : Function = prim::Constant[name="layer_norm"]()
  %9 : int = prim::Constant[value=1]()
  %x.1 : Tensor = aten::add(%input.1, %bias.1, %9) # fuse_add_layer_norm_pass.py:15:8
  %x.5 : Tensor = aten::add(%x.1, %residual.1, %9) # fuse_add_layer_norm_pass.py:16:8
  %20 : float = aten::FloatImplicit(%normalized_eps.1) # fuse_add_layer_norm_pass.py:17:11
  %22 : Tensor = prim::CallFunction(%21, %x.5, %normalized_shape.1, %normalized_weight.1, %normalized_bias.1, %20) # fuse_add_layer_norm_pass.py:17:11
  return (%22)
"""

fuse = """
graph(%input.1 : Tensor,
      %residual.1 : Tensor,
      %bias.1 : Tensor,
      %normalized_shape : int[],
      %normalized_weight.1 : Tensor,
      %normalized_bias.1 : Tensor,
      %normalized_eps : Tensor):
  %12 : Tensor = fastseq::fused_add_layer_norm(%input.1, %residual.1, %normalized_weight.1, %normalized_bias.1, %bias.1) # fuse_add_layer_norm_pass.py:27:13
  return (%12)
"""

input = torch.ones([16, 768]).cuda()
residual = torch.ones([16, 768]).cuda()
bias = torch.ones([768]).cuda()
normalized_shape = [768]
normalized_weight = torch.ones([768]).cuda()
normalized_bias = torch.ones([768]).cuda()
normalized_eps = 1e-05

torch.cuda.synchronize()
t1 = time.time()
for _ in range(100):
  expected_output = _add_layer_norm(input, residual, bias, normalized_shape, normalized_weight, normalized_bias, normalized_eps)
torch.cuda.synchronize()
t2 = time.time()
print(f"_add_layer_norm: {(t2 - t1)/100}")

torch.cuda.synchronize()
t1 = time.time()
for _ in range(100):
  generated_output = _fused_add_layer_norm(input, residual, bias, normalized_shape, normalized_weight, normalized_bias, normalized_eps)
torch.cuda.synchronize()
t2 = time.time()
print(f"_fused_add_layer_norm: {(t2 - t1)/100}")

import fastseq_extension

script_func = torch.jit.script(_add_layer_norm)



for _ in range(100):
    script_func(input, residual, bias, normalized_shape, normalized_weight, normalized_bias, normalized_eps)

print("+++++++++++++++++++ Optimized graph: \n")


# fastseq_extension._jit_fuse_layer_norm(script_func.graph)
# torch._C._jit_pass_fuse_linear(script_func.graph)

print(script_func.graph_for(input, residual, bias, normalized_shape, normalized_weight, normalized_bias, normalized_eps))

print(script_func.graph)

torch.cuda.synchronize()
t1 = time.time()
for _ in range(100):
    script_func(input, residual, bias, normalized_shape, normalized_weight, normalized_bias, normalized_eps)
torch.cuda.synchronize()
t2 = time.time()

print(f"time:  {(t2 - t1)/100}")
