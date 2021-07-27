import torch
import time

# print(input)
torch.classes.load_library("build/lib/libpyt_fastertransformer.so")

# warm up
# input = torch.rand((128, 16), device='cuda')
# values, ids = torch.topk(input, 1)
# sort_ids = torch.ops.fastertransformer.topk(input, 1)


K = 4
input = torch.rand([32, 2, 8192], device='cuda')
REPEAT_NUM = 1


torch.cuda.synchronize()
t1 = time.time()
# sort_ids = [torch.ops.fastertransformer.topk(input, K) for _ in range(REPEAT_NUM)]
torch.cuda.synchronize()
t2 = time.time()

torch.cuda.synchronize()
t3 = time.time()
sort_ids = [torch.ops.fastseq.topk_v2(input, K) for _ in range(REPEAT_NUM)]
torch.cuda.synchronize()
t4 = time.time()

torch.cuda.synchronize()
t5 = time.time()
value_ids = [torch.topk(input, K) for _ in range(REPEAT_NUM)]
torch.cuda.synchronize()
t6 = time.time()

print(f"tf-v1, tf-v2, pytorch: {(t2 - t1) / (t6 - t5)}, {(t4 - t3) / (t6 - t5)}; \n")

print(f"expected torch.topk: \n{value_ids[0][0]}, \n{value_ids[0][1]}")
print(f"result: \n{sort_ids[0]}")
# print(input)
