import time
import torch

torch.classes.load_library("lib/libpyt_fastertransformer.so")

d0 = 768*4
d1 = 768
batch_size = 16
input = torch.ones([batch_size, 1, d0]).cuda()
resident = torch.ones([batch_size, 1, d1]).cuda()

class Base(torch.nn.Module):
  def __init__(self):
    super(Base, self).__init__()

    self.linear = torch.nn.Linear(d0, d1)
    self.layer_norm = torch.nn.LayerNorm(d1).cuda()
  
  def forward(self, input, resident):
    y = self.linear(input)
    y = y + resident
    y = self.layer_norm(y)
    return y

base = Base().cuda()

scripted_func = torch.jit.script(base)

print(scripted_func.graph)

torch.cuda.synchronize()
torch.cuda.nvtx.range_push("torchscript")
expected_output = scripted_func(input, resident)
torch.cuda.nvtx.range_pop()

# linear_weigth = base.linear.weight.transpose(0, 1).contiguous()

print(base.linear.weight.shape)
print(input.shape)

torch.cuda.synchronize()
torch.cuda.nvtx.range_push("fastseq")
output = torch.ops.fastseq.fused_add_layer_norm(
  input, resident, base.linear.weight, base.linear.bias, base.layer_norm.weight, base.layer_norm.bias)
torch.cuda.nvtx.range_pop()


print(expected_output)
print(output)

TIMES = 100

#warm up
for _ in range(TIMES):
  expected_output = scripted_func(input, resident)
for _ in range(TIMES):
  output = torch.ops.fastseq.fused_add_layer_norm(input, resident, base.linear.weight, base.linear.bias, base.layer_norm.weight, base.layer_norm.bias)

torch.cuda.synchronize()
t1 = time.time()
for _ in range(TIMES):
  expected_output = scripted_func(input, resident)
torch.cuda.synchronize()
t2 = time.time()
print(f"torchscript: {(t2 - t1)/TIMES}")


torch.cuda.synchronize()
t3 = time.time()
for _ in range(TIMES):
  output = torch.ops.fastseq.fused_add_layer_norm(input, resident, base.linear.weight, base.linear.bias, base.layer_norm.weight, base.layer_norm.bias)
torch.cuda.synchronize()
t4 = time.time()
print(f"fastseq: {(t4 - t3)/TIMES}")




print(f"torchscript v.s. fastseq: {(t2 - t1)/(t4 - t3)}")