import torch
import time

import os

print(f"****** {os.getpid()} \n")
# print(input)
torch.classes.load_library("/datadrive/fhu/github/FasterTransformer/build_15/lib/libpyt_fastertransformer.so")
# torch.classes.load_library("build/lib/libpyt_fastertransformer.so")

# warm up
# input = torch.rand((128, 16), device='cuda')
# values, ids = torch.topk(input, 1)
# sort_ids = torch.ops.fastertransformer.topk(input, 1)


K = 3
input = torch.rand([18, 29824], device='cuda')
REPEAT_NUM = 1000

# warmup
expected_value_ids = [torch.topk(input, K) for _ in range(3)]
value_ids = [torch.ops.fastseq.topk(input, K) for _ in range(3)]

torch.cuda.synchronize()
t3 = time.time()
value_ids = [torch.ops.fastseq.topk(input, K) for _ in range(REPEAT_NUM)]
torch.cuda.synchronize()
t4 = time.time()
print(t4 - t3)

torch.cuda.synchronize()
t5 = time.time()
expected_value_ids = [torch.topk(input, K) for _ in range(REPEAT_NUM)]
torch.cuda.synchronize()
t6 = time.time()

print(t6 - t5)

# print(f"expected torch.topk: \n{expected_value_ids[0][0]}, \n{expected_value_ids[0][1]}")
# print(f"result: \n{value_ids[0][0]}, \n{value_ids[0][1]}")
assert (expected_value_ids[0][0] == value_ids[0][0]).all() or (expected_value_ids[0][1] == value_ids[0][1]).all()
# print(input)

print(f"ft-v2, pytorch: {(t4 - t3) / (t6 - t5)}; \n")
