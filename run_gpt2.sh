#!/bin/bash

# git clone git@github.com:feihugis/FasterTransformer.git
mkdir build
cd build
git submodule init && git submodule update
# cmake -DSM=90 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DENABLE_FP8=ON ..
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..

make -j
pip install -r ../examples/pytorch/gpt/requirement.txt

cd ../
git clone https://huggingface.co/gpt2-xl
python examples/pytorch/gpt/utils/huggingface_gpt_convert.py -i gpt2-xl/ -o ./models/huggingface-models/c-model/gpt2-xl -i_g 1

wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
mkdir models/345m/ -p
unzip megatron_lm_345m_v0.0.zip -d ./models/345m

export PYTHONPATH=$PWD/..:${PYTHONPATH}
python3 examples/pytorch/gpt/utils/megatron_fp8_ckpt_convert.py \
      -i ./models/345m/release \
      -o ./models/345m/c-model/ \
      -i_g 1 \
      -head_num 16 \
      -trained_tensor_parallel_size 1

python3 examples/pytorch/gpt/gpt_summarization.py \
        --data_type fp16 \
        --lib_path ./build/lib/libth_transformer.so \
        --summarize \
        --ft_model_location ./models/345m/c-model/ \
        --hf_model_location ./gpt2-xl/

python examples/pytorch/gpt/multi_gpu_gpt_example.py \
            --data_type fp16 \
            --lib_path ./build/lib/libth_transformer.so \
            --ckpt_path ./models/345m/c-model/1-gpu/ \
            --vocab_file ./gpt2-xl/vocab.json \
            --merges_file ./gpt2-xl/merges.txt \
            --time \
            --input_len 16 \
            --output_len 8 \
            --beam_width 8 \
            --len_penalty 1.0 \
            --max_batch_size 4 \
            --repetition_penalty 1.0
