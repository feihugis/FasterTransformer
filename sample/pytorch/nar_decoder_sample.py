import time
import torch

from onmt.utils.misc import sequence_mask
from utils.decoder import CustomDecoder, ONMTDecoder, init_op_cache, init_onmt_cache

import os
print(f"pid: {os.getpid()} \n")

torch.set_printoptions(linewidth=400, edgeitems=3, precision=6, profile="default")
class DecoderWeights(object):
    def __init__(self, layer_num, hidden_dim, path='/model/model_checkpoint/checkpoint_best.pt'):
        self.layer_num = layer_num
        self.w = [[] for _ in range(layer_num)]

        if path is not None and layer_num == 6 and hidden_dim == 512:
            model = torch.load(path)
            model = model['model']
            for i, layer_weights in enumerate(self.w):
                pref = f"decoder.layers.{i}."
                layer_weights.append(model[pref + "self_attn_layer_norm.weight"])   # self_layernorm_gamma
                layer_weights.append(model[pref + "self_attn_layer_norm.bias"])   # self_layernorm_beta
                layer_weights.append(model[pref + "self_attn.q_proj.weight"].transpose(0, 1).contiguous())   # self_kernel_q
                layer_weights.append(model[pref + "self_attn.k_proj.weight"].transpose(0, 1).contiguous())   # self_kernel_k
                layer_weights.append(model[pref + "self_attn.v_proj.weight"].transpose(0, 1).contiguous())   # self_kernel_v
                layer_weights.append(model[pref + "self_attn.q_proj.bias"])   # self_bias_q
                layer_weights.append(model[pref + "self_attn.k_proj.bias"])   # self_bias_k
                layer_weights.append(model[pref + "self_attn.v_proj.bias"])   # self_bias_v
                layer_weights.append(model[pref + "self_attn.out_proj.weight"].transpose(0, 1).contiguous())   # self_output_kernel
                layer_weights.append(model[pref + "self_attn.out_proj.bias"])   # self_output_bias
                layer_weights.append(model[pref + "encoder_attn_layer_norm.weight"])   # cross_layernorm_gamma
                layer_weights.append(model[pref + "encoder_attn_layer_norm.bias"])   # cross_layernorm_beta
                layer_weights.append(model[pref + "encoder_attn.q_proj.weight"].transpose(0, 1).contiguous())   # cross_kernel_q
                layer_weights.append(model[pref + "encoder_attn.k_proj.weight"].transpose(0, 1).contiguous())   # cross_kernel_k
                layer_weights.append(model[pref + "encoder_attn.v_proj.weight"].transpose(0, 1).contiguous())   # cross_kernel_v
                layer_weights.append(model[pref + "encoder_attn.q_proj.bias"])   # cross_bias_q
                layer_weights.append(model[pref + "encoder_attn.k_proj.bias"])   # cross_bias_k
                layer_weights.append(model[pref + "encoder_attn.v_proj.bias"])   # cross_bias_v
                layer_weights.append(model[pref + "encoder_attn.out_proj.weight"].transpose(0, 1).contiguous())   # cross_output_kernel
                layer_weights.append(model[pref + "encoder_attn.out_proj.bias"])   # cross_output_bias
                layer_weights.append(model[pref + "final_layer_norm.weight"])   # ffn_layernorm_gamma
                layer_weights.append(model[pref + "final_layer_norm.bias"])   # ffn_layernorm_beta
                layer_weights.append(model[pref + "fc1.weight"].transpose(0, 1).contiguous())   # inter_kernel
                layer_weights.append(model[pref + "fc1.bias"])   # inter_bias
                layer_weights.append(model[pref + "fc2.weight"].transpose(0, 1).contiguous())   # output_kernel
                layer_weights.append(model[pref + "fc2.bias"])   # output_bias
                # for i in range(len(layer_weights)):
                #     torch.nn.init.uniform_(layer_weights[i], -1, 1)
        
        else:
            for layer_weights in self.w:
                layer_weights.append(torch.zeros(hidden_dim))   # self_layernorm_gamma
                layer_weights.append(torch.zeros(hidden_dim))   # self_layernorm_beta
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # self_kernel_q
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # self_kernel_k
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # self_kernel_v
                layer_weights.append(torch.zeros(hidden_dim))   # self_bias_q
                layer_weights.append(torch.zeros(hidden_dim))   # self_bias_k
                layer_weights.append(torch.zeros(hidden_dim))   # self_bias_v
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # self_output_kernel
                layer_weights.append(torch.zeros(hidden_dim))   # self_output_bias
                layer_weights.append(torch.zeros(hidden_dim))   # cross_layernorm_gamma
                layer_weights.append(torch.zeros(hidden_dim))   # cross_layernorm_beta
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_kernel_q
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_kernel_k
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_kernel_v
                layer_weights.append(torch.zeros(hidden_dim))   # cross_bias_q
                layer_weights.append(torch.zeros(hidden_dim))   # cross_bias_k
                layer_weights.append(torch.zeros(hidden_dim))   # cross_bias_v
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_output_kernel
                layer_weights.append(torch.zeros(hidden_dim))   # cross_output_bias
                layer_weights.append(torch.zeros(hidden_dim))   # ffn_layernorm_gamma
                layer_weights.append(torch.zeros(hidden_dim))   # ffn_layernorm_beta
                layer_weights.append(torch.zeros(hidden_dim, 4 * hidden_dim))   # inter_kernel
                layer_weights.append(torch.zeros(4 * hidden_dim))   # inter_bias
                layer_weights.append(torch.zeros(4 * hidden_dim, hidden_dim))   # output_kernel
                layer_weights.append(torch.zeros(hidden_dim))   # output_bias
                for i in range(len(layer_weights)):
                    torch.nn.init.uniform_(layer_weights[i], -1, 1)

    def to_cuda(self):
        for i in range(self.layer_num):
            for j in range(len(self.w[i])):
                self.w[i][j] = self.w[i][j].cuda()

    def to_half(self):
        for i in range(self.layer_num):
            for j in range(len(self.w[i])):
                self.w[i][j] = self.w[i][j].half()

def run():
    layer_num = 6
    head_num = 8
    head_size = 64
    hidden_dim = 512
    batch_size = 1
    beam_width = 1
    mem_seq_len = 16
    max_seq_len = 16
    decoding_max_seq_len = 32

    step = 1
    path = '/model/model_checkpoint/checkpoint_best.pt'
    # path = None
    is_fp16 = False

    mem_seq_lens = torch.randint(mem_seq_len, mem_seq_len + 1, (batch_size,), dtype=torch.int32).cuda()
    src_pad_mask = ~sequence_mask(mem_seq_lens, mem_seq_len).unsqueeze(1)

    weights = DecoderWeights(layer_num, hidden_dim, path)

    onmt_decoder = ONMTDecoder(layer_num, head_num, head_size, weights)
    onmt_decoder.cuda()
    onmt_decoder.eval()

    weights.to_cuda()
    custom_decoder = CustomDecoder(layer_num, head_num, head_size, hidden_dim, weights, is_fp16)


    with torch.no_grad():
        
        self_cache, mem_cache = init_op_cache(layer_num, batch_size, beam_width, max_seq_len, decoding_max_seq_len, head_num, head_size, hidden_dim, is_fp16)
        
        input_seq_len = 6
        for input_seq_len in range(1, 17, 1):
            print(f"\n------------ Test input_seq_len = {input_seq_len}")
            inp = torch.empty(batch_size, input_seq_len, hidden_dim).cuda()
            mem = torch.empty(batch_size, mem_seq_len, hidden_dim).cuda()
            torch.nn.init.uniform_(inp, -1, 1)
            torch.nn.init.uniform_(mem, -1, 1)

            cache = init_onmt_cache(layer_num, mem)

            torch.cuda.synchronize()
            t1 = time.time()
            ft_output, self_cache, mem_cache = custom_decoder(inp, mem, input_seq_len, mem_seq_len, self_cache, mem_cache, 1)
            torch.cuda.synchronize()
            t2 = time.time()
            print(f"FT time: {t2 - t1}: {ft_output.shape}")
            
            torch.cuda.synchronize()
            t3 = time.time()
            onmt_output = onmt_decoder(inp, mem, src_pad_mask, cache, 0)
            torch.cuda.synchronize()
            t4 = time.time()
            print(f"ONMT time: {t4 - t3}: {onmt_output.shape}")

            diff = torch.abs(ft_output - onmt_output)
            print('Mean diff: {}'.format(torch.mean(diff)))
            print('Max diff:  {}'.format(torch.max(diff)))
            print('Min diff:  {}'.format(torch.min(diff)))


if __name__ == '__main__':
    run()
