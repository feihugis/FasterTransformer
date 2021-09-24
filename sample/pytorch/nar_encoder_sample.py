from __future__ import print_function

import os
import sys
import torch
import time

from transformers import BertConfig
from transformers.modeling_bert import BertEncoder

try:
    from utils.ckpt_quantization import checkpoint_quantization
    from utils.encoder import EncoderWeights, CustomEncoder
except:
    from ckpt_quantization import checkpoint_quantization
    from encoder import EncoderWeights, CustomEncoder

class NarEncoderWeights(object):
    def __init__(self, layer_num, hidden_dim, weights=None):
        """weights need be a state_dict of bert model"""
        self.layer_num = layer_num
        self.int8 = False
        self.hidden_dim = hidden_dim
        self.weights = {}
        self.hf_weights = {}
        self.nar_2_raw = {}
        self.raw_2_nar = {}
        
        for i in range(layer_num):
          self.nar_2_raw[f"encoder.layers.{i}.self_attn.q_proj.weight"] = f"bert.encoder.layer.{i}.attention.self.query.weight"
          self.nar_2_raw[f"encoder.layers.{i}.self_attn.q_proj.bias"] = f"bert.encoder.layer.{i}.attention.self.query.bias"
          self.nar_2_raw[f"encoder.layers.{i}.self_attn.k_proj.weight"] = f"bert.encoder.layer.{i}.attention.self.key.weight"
          self.nar_2_raw[f"encoder.layers.{i}.self_attn.k_proj.bias"] = f"bert.encoder.layer.{i}.attention.self.key.bias"
          self.nar_2_raw[f"encoder.layers.{i}.self_attn.v_proj.weight"] = f"bert.encoder.layer.{i}.attention.self.value.weight"
          self.nar_2_raw[f"encoder.layers.{i}.self_attn.v_proj.bias"] = f"bert.encoder.layer.{i}.attention.self.value.bias"
          self.nar_2_raw[f"encoder.layers.{i}.self_attn.out_proj.weight"] = f"bert.encoder.layer.{i}.attention.output.dense.weight"
          self.nar_2_raw[f"encoder.layers.{i}.self_attn.out_proj.bias"] = f"bert.encoder.layer.{i}.attention.output.dense.bias"
          self.nar_2_raw[f"encoder.layers.{i}.self_attn_layer_norm.weight"] = f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight"
          self.nar_2_raw[f"encoder.layers.{i}.self_attn_layer_norm.bias"] = f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias"
          self.nar_2_raw[f"encoder.layers.{i}.fc1.weight"] = f"bert.encoder.layer.{i}.intermediate.dense.weight"
          self.nar_2_raw[f"encoder.layers.{i}.fc1.bias"] = f"bert.encoder.layer.{i}.intermediate.dense.bias"
          self.nar_2_raw[f"encoder.layers.{i}.fc2.weight"] = f"bert.encoder.layer.{i}.output.dense.weight"
          self.nar_2_raw[f"encoder.layers.{i}.fc2.bias"] = f"bert.encoder.layer.{i}.output.dense.bias"
          self.nar_2_raw[f"encoder.layers.{i}.final_layer_norm.weight"] = f"bert.encoder.layer.{i}.output.LayerNorm.weight"
          self.nar_2_raw[f"encoder.layers.{i}.final_layer_norm.bias"] = f"bert.encoder.layer.{i}.output.LayerNorm.bias"

          self.raw_2_nar[f"bert.encoder.layer.{i}.attention.self.query.weight"] = f"encoder.layers.{i}.self_attn.q_proj.weight"
          self.raw_2_nar[f"bert.encoder.layer.{i}.attention.self.query.bias"]  = f"encoder.layers.{i}.self_attn.q_proj.bias"
          self.raw_2_nar[f"bert.encoder.layer.{i}.attention.self.key.weight"] = f"encoder.layers.{i}.self_attn.k_proj.weight"
          self.raw_2_nar[f"bert.encoder.layer.{i}.attention.self.key.bias"] = f"encoder.layers.{i}.self_attn.k_proj.bias"
          self.raw_2_nar[f"bert.encoder.layer.{i}.attention.self.value.weight"] = f"encoder.layers.{i}.self_attn.v_proj.weight"
          self.raw_2_nar[f"bert.encoder.layer.{i}.attention.self.value.bias"] = f"encoder.layers.{i}.self_attn.v_proj.bias"
          self.raw_2_nar[f"bert.encoder.layer.{i}.attention.output.dense.weight"] = f"encoder.layers.{i}.self_attn.out_proj.weight"
          self.raw_2_nar[f"bert.encoder.layer.{i}.attention.output.dense.bias"] = f"encoder.layers.{i}.self_attn.out_proj.bias"
          self.raw_2_nar[f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight"] = f"encoder.layers.{i}.self_attn_layer_norm.weight"
          self.raw_2_nar[f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias"] = f"encoder.layers.{i}.self_attn_layer_norm.bias"
          self.raw_2_nar[f"bert.encoder.layer.{i}.intermediate.dense.weight"] = f"encoder.layers.{i}.fc1.weight"
          self.raw_2_nar[f"bert.encoder.layer.{i}.intermediate.dense.bias"] = f"encoder.layers.{i}.fc1.bias"
          self.raw_2_nar[f"bert.encoder.layer.{i}.output.dense.weight"] = f"encoder.layers.{i}.fc2.weight"
          self.raw_2_nar[f"bert.encoder.layer.{i}.output.dense.bias"] = f"encoder.layers.{i}.fc2.bias"
          self.raw_2_nar[f"bert.encoder.layer.{i}.output.LayerNorm.weight"] = f"encoder.layers.{i}.final_layer_norm.weight"
          self.raw_2_nar[f"bert.encoder.layer.{i}.output.LayerNorm.bias"] = f"encoder.layers.{i}.final_layer_norm.bias"


        
        if weights is None:
            self._generated_weights = True
            for i in range(layer_num):
                pre = 'bert.encoder.layer.' + str(i) + '.'
                self.weights[pre + 'attention.self.query.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.self.query.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.self.key.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.self.key.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.self.value.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.self.value.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.output.dense.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.output.dense.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.output.LayerNorm.weight'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.output.LayerNorm.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'intermediate.dense.weight'] = torch.zeros(4 * hidden_dim, hidden_dim)
                self.weights[pre + 'intermediate.dense.bias'] = torch.zeros(4 * hidden_dim)
                self.weights[pre + 'output.dense.weight'] = torch.zeros(hidden_dim, 4 * hidden_dim)
                self.weights[pre + 'output.dense.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'output.LayerNorm.weight'] = torch.zeros(hidden_dim)
                self.weights[pre + 'output.LayerNorm.bias'] = torch.zeros(hidden_dim)
            for k, v in self.weights.items():
                if not k.endswith('_amax'):
                    self.weights[k] = torch.nn.init.uniform_(v, -40, 40)
            self.hf_weights = self.weights
        else:
            self._generated_weights = False
 
            for k, v in weights.items():
                # if ('fc1' in k):
                #     print("+++", k, v.min(), v.max(), v.dtype)
                #     v = torch.nn.init.uniform_(v, -1, 1)
                ks = k.split('.')
                if ks[0] == 'encoder':
                    self.weights[k] = v
                    if k in self.nar_2_raw:
                        self.hf_weights[self.nar_2_raw[k]] = v
                if ks[0] == 'bert':
                    self.weights[self.raw_2_nar[k]] = v
                    self.hf_weights[k] = v

    def listed_weights(self, layer_idx):
        ret = []
        
        if self._generated_weights:
            pre = 'bert.encoder.layer.' + str(layer_idx) + '.'
            ret.append(self.weights[pre + 'attention.self.query.weight'])       # 0
            ret.append(self.weights[pre + 'attention.self.query.bias'])
            ret.append(self.weights[pre + 'attention.self.key.weight'])         # 2
            ret.append(self.weights[pre + 'attention.self.key.bias'])
            ret.append(self.weights[pre + 'attention.self.value.weight'])       # 4
            ret.append(self.weights[pre + 'attention.self.value.bias'])
            ret.append(self.weights[pre + 'attention.output.dense.weight'])     # 6
            ret.append(self.weights[pre + 'attention.output.dense.bias'])
            ret.append(self.weights[pre + 'attention.output.LayerNorm.weight'])
            ret.append(self.weights[pre + 'attention.output.LayerNorm.bias'])
            ret.append(self.weights[pre + 'intermediate.dense.weight'])         # 10
            ret.append(self.weights[pre + 'intermediate.dense.bias'])
            ret.append(self.weights[pre + 'output.dense.weight'])               # 12
            ret.append(self.weights[pre + 'output.dense.bias'])
            ret.append(self.weights[pre + 'output.LayerNorm.weight'])
            ret.append(self.weights[pre + 'output.LayerNorm.bias'])
        else:
            pre = 'encoder.layers.' + str(layer_idx) + '.'
            ret.append(self.weights[pre + 'self_attn.q_proj.weight'])
            ret.append(self.weights[pre + 'self_attn.q_proj.bias'])
            ret.append(self.weights[pre + 'self_attn.k_proj.weight'])
            ret.append(self.weights[pre + 'self_attn.k_proj.bias'])
            ret.append(self.weights[pre + 'self_attn.v_proj.weight'])
            ret.append(self.weights[pre + 'self_attn.v_proj.bias'])
            ret.append(self.weights[pre + 'self_attn.out_proj.weight'])
            ret.append(self.weights[pre + 'self_attn.out_proj.bias'])
            ret.append(self.weights[pre + 'self_attn_layer_norm.weight'])
            ret.append(self.weights[pre + 'self_attn_layer_norm.bias'])
            ret.append(self.weights[pre + 'fc1.weight'])
            ret.append(self.weights[pre + 'fc1.bias'])
            ret.append(self.weights[pre + 'fc2.weight'])
            ret.append(self.weights[pre + 'fc2.bias'])
            ret.append(self.weights[pre + 'final_layer_norm.weight'])
            ret.append(self.weights[pre + 'final_layer_norm.bias'])


        if not self.int8:
            ret[0] = ret[0].transpose(-1, -2).contiguous()
            ret[2] = ret[2].transpose(-1, -2).contiguous()
            ret[4] = ret[4].transpose(-1, -2).contiguous()
            ret[6] = ret[6].transpose(-1, -2).contiguous()
            ret[10] = ret[10].transpose(-1, -2).contiguous()
            ret[12] = ret[12].transpose(-1, -2).contiguous()
            ret.append(torch.tensor(0))
        else:
            ret.append(self.weights[pre + 'amaxList'])
        return ret

    def to_cuda(self):
        for k, v in self.weights.items():
            self.weights[k] = v.cuda()

        for k, v in self.hf_weights.items():
            self.hf_weights[k] = v.cuda()

    def to_half(self):
        if self.int8:
            raise RuntimeError("Cannot cast to half if the weights have been casted to int8.")
        for k, v in self.weights.items():
            self.weights[k] = v.half()

    def to_int8(self, is_per_channel, ths_path='./lib/libpyt_fastertransformer.so'):
        if self._generated_weights:
            if is_per_channel:
                amax_tensor_1 = torch.Tensor(self.hidden_dim).fill_(127.)
                amax_tensor_2 = torch.Tensor(self.hidden_dim * 4).fill_(127.)
            else:
                amax_tensor_1 = torch.tensor(127.)
                amax_tensor_2 = torch.tensor(127.)
            for i in range(self.layer_num):
                pre = 'bert.encoder.layer.' + str(i) + '.'
                self.weights[pre + 'attention.self.query._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.query._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.self.query._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.key._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.key._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.self.key._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.value._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.value._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.self.value._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_q_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_k_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_v_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_a_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.softmax_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.dense._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.dense._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.output.dense._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.add_local_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.add_residual_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'intermediate.dense._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'intermediate.dense._weight_quantizer._amax'] = amax_tensor_2
                self.weights[pre + 'intermediate.dense._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.dense._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.dense._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'output.dense._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.add_local_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.add_residual_input_quantizer._amax'] = torch.tensor(127.)
        if 'bert.encoder.layer.0.attention.self.query._input_quantizer._amax' not in self.weights:
            raise RuntimeError("There is no quantization node in the checkpoint, cannot be quantized to int8.")
        if self.int8:
            return
        self.int8 = True
        for k, v in self.weights.items():
            if k.endswith('bias') or k.endswith('LayerNorm.weight'):
                self.weights[k] = v.half()
            else:
                self.weights[k] = v.float().cpu()
        self.weights = checkpoint_quantization(self.weights, is_per_channel, ths_path, verbose=False)

class HuggingFaceEncoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights=None):
        super().__init__()
        hidden_dim = head_num * head_size
        conf = BertConfig(
            hidden_size=hidden_dim,
            intermediate_size=4*hidden_dim,
            num_attention_heads=head_num,
            num_hidden_layers=layer_num,
            hidden_act='relu')

        self.encoder = BertEncoder(conf)
        w = {}
        for k, v in weights.hf_weights.items():
            if k.startswith('bert.encoder') and not k.endswith('_amax'):
                w[k[13:]] = weights.hf_weights[k]
        self.encoder.load_state_dict(w)
        self.head_mask = [None] * layer_num

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        output = self.encoder(hidden_states, extended_attention_mask, self.head_mask)
        return output

def sequence_mask(lengths, max_len=None, is_2d=True):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    if is_2d:
        return mask
    else:
        mask = mask.view(-1, 1, 1, max_len)
        m2 = mask.transpose(2, 3)
        return mask * m2

def run():

    batch_size = 1
    seq_len = 6
    layer_num=6
    hidden_dim=512

    inp = torch.empty(batch_size, seq_len, hidden_dim).cuda()
    torch.nn.init.uniform_(inp, -1, 1)
    mem_seq_lens = torch.randint(seq_len, seq_len+1, (batch_size,), dtype=torch.int32).cuda()
    mask = sequence_mask(mem_seq_lens, seq_len, False).to(torch.float)

    model = torch.load('/model/model_checkpoint/checkpoint_best.pt')
    weights = NarEncoderWeights(layer_num=6, hidden_dim=512, weights=model['model'])

    # model = torch.load('/datadrive/fhu/github/FasterTransformer/build_15/test.pt')
    # model = torch.load('/datadrive/fhu/github/FasterTransformer/build_15/hf_test.pt')
    # model = torch.load('/datadrive/fhu/github/FasterTransformer/build_15/nar_test.pt')
    # weights = NarEncoderWeights(layer_num=6, hidden_dim=512, weights=model)

    # weights = NarEncoderWeights(layer_num=6, hidden_dim=512, weights=None)

    # torch.save(weights.weights, 'nar_test.pt')

    # torch.save(weights.hf_weights, 'hf_test.pt')

    weights.to_cuda()

    custom_encoder = CustomEncoder(
    layer_num=6,
    head_num=8,
    head_size=64,
    weights=weights,
    int8_mode=0,
    remove_padding=False,
    allow_gemm_test=True,
    path='lib/libpyt_fastertransformer.so')
    # custom_encoder = torch.jit.script(custom_encoder)

    hf_encoder = HuggingFaceEncoder(
        layer_num=6,
        head_num=8,
        head_size=64,
        weights=weights
    )
    hf_encoder.cuda()
    hf_encoder.eval()
    # hf_encoder = torch.jit.trace(hf_encoder, (inp, mask))

    print(f"******************* current pid:{os.getpid()} \n")

    with torch.no_grad():
        output_mask = sequence_mask(mem_seq_lens, seq_len).to(mask.dtype).unsqueeze(-1)
        
        torch.cuda.synchronize()
        t1 = time.time()
        ft_output = custom_encoder(inp, mask, mem_seq_lens)[0] #* output_mask
        torch.cuda.synchronize()
        t2 = time.time()
        print(f"====== ft_output: {t2 - t1}\n", ft_output)
        
        torch.cuda.synchronize()
        t3 = time.time()
        hf_output = hf_encoder(inp, mask)[0] #* output_mask
        torch.cuda.synchronize()
        t4 = time.time()
        print(f"====== hf_output: {t4 - t3}\n", hf_output)

        
        inp = torch.empty(batch_size, seq_len, hidden_dim).cuda()
        torch.nn.init.uniform_(inp, -1, 1)
        mem_seq_lens = torch.randint(seq_len, seq_len+1, (batch_size,), dtype=torch.int32).cuda()
        mask = sequence_mask(mem_seq_lens, seq_len, False).to(torch.float)

        torch.cuda.synchronize()
        t1 = time.time()
        ft_output = custom_encoder(inp, mask, mem_seq_lens)[0] #* output_mask
        torch.cuda.synchronize()
        t2 = time.time()
        print(f"====== ft_output: {t2 - t1}\n", ft_output)
        
        torch.cuda.synchronize()
        t3 = time.time()
        hf_output = hf_encoder(inp, mask)[0] #* output_mask
        torch.cuda.synchronize()
        t4 = time.time()
        print(f"====== hf_output: {t4 - t3}\n", hf_output)

        diff = torch.abs(hf_output - ft_output)
        print('Mean diff: {}'.format(torch.mean(diff)))
        print('Max diff:  {}'.format(torch.max(diff)))
        print('Min diff:  {}'.format(torch.min(diff)))

if __name__ == '__main__':
    run()
