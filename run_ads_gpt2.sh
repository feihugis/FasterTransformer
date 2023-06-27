#!/bin/bash

conda activate pytorch

python ./examples/onnx/multi_gpu_gpt/onnx_ckpt_convert.py \
  -i ./examples/pytorch/gpt/ads_data/model.onnx \
  -o ./models/ads-gpt2-models/c-model/ \
  -i_g 1

# decoder: keys = [v.name for v in model.graph.node[0].attribute[6].g.initializer]
# init_decoder: [v.name for v in model.graph.node[0].attribute[5].g.initializer]

export FT_NVTX=ON
export CUDA_VISIBLE_DEVICES=MIG-73ea7899-43fa-5c14-98d1-67eea628afd0
python examples/pytorch/gpt/ads_gpt2.py \
            --data_type fp16 \
            --lib_path ./build_mig/lib/libth_transformer.so \
            --ckpt_path ./models/ads-gpt2-models/c-model/1-gpu/ \
            --vocab_file ./gpt2/vocab.json \
            --merges_file ./gpt2/merges.txt \
            --time \
            --input_len 6 \
            --output_len 8 \
            --beam_width 8 \
            --len_penalty 1.0 \
            --max_batch_size 5 \
            --repetition_penalty 1.0 \
            --size_per_head 32 \
            --head_num 8 \
            --layer_num 3 \
            --vocab_size 50110 \
            --max_seq_len 64 \
            --start_id 1 \
            --end_id 2 \
            --presence_penalty 2.0
