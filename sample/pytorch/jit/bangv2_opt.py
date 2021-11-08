import numpy as np
import time
import torch

torch.classes.load_library("lib/libpyt_fastertransformer.so")

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)
torch._C._jit_set_bailout_depth(0)

model = torch.jit.load("/model/bang_v2_torchscript_opt_test.pt")

with torch.no_grad():
  static_src_tokens = torch.LongTensor(np.array([[101, 2040, 102, 101, 2003, 102, 101, 3419, 102, 101, 8592, 102, 102, 0, 0, 0]])).cuda()
  static_src_tokens.requires_grad_(False)
  static_src_lens = torch.tensor([16], device='cuda', requires_grad=False)
  static_prev_out_tokens = static_src_tokens.new_ones([1, 16], device='cuda', requires_grad=False)



# g = model.graph_for(static_src_tokens, static_src_lens, static_prev_out_tokens)

# print(model.graph)

# for _ in range(10):
#     output = model(static_src_tokens, static_src_lens, static_prev_out_tokens)

# print(model.graph)

import fastseq_extension
fastseq_extension._jit_fuse_layer_norm(model.graph)
model.save("graph_opt_test.pt")
model = torch.jit.load("graph_opt_test.pt")

torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_texpr_fuser_enabled(True)
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_bailout_depth(1)

print(model.code)
print(model.graph)

for _ in range(10):
    output = model(static_src_tokens, static_src_lens, static_prev_out_tokens)


torch.cuda.synchronize()
t1 = time.time()
with torch.no_grad():
  for _ in range(10):
      output = model(static_src_tokens, static_src_lens, static_prev_out_tokens)
torch.cuda.synchronize()
t2 = time.time()
print(f"torchscript time: {(t2 - t1) / 100}")

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_src_tokens, static_src_lens, static_prev_out_tokens)

for _ in range(10):
  g.replay()

torch.cuda.synchronize()
t3 = time.time()
for _ in range(10):
  g.replay()
torch.cuda.synchronize()
t4 = time.time()
print(f"cuda graph time: {(t4 - t3) / 100}") 