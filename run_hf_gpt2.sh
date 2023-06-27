# Description: Run HF GPT2 model with FasterTransformer

git clone git@github.com:feihugis/FasterTransformer.git
mkdir build
cd build
git submodule init && git submodule update
# cmake -DSM=90 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DENABLE_FP8=ON ..
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Debug -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..


python ./examples/onnx/multi_gpu_gpt/onnx_ckpt_convert.py -i ./onnx_models/gpt2_past.onnx -o ./models/onnx-models/c-model/124m-v2/ -i_g 1
python ./examples/onnx/multi_gpu_gpt/onnx_ckpt_convert.py -i ./hf_gpt2_medium/decoder_model.onnx -o ./models/onnx-models/c-model/355m/ -i_g 1

git clone https://huggingface.co/gpt2
wget https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-10.onnx -P ./models
python ./examples/onnx/multi_gpu_gpt/onnx_ckpt_convert.py -i ./models/gpt2-10.onnx -o ./models/onnx-models/c-model/124m/ -i_g 1

export FT_NVTX=ON
export CUDA_VISIBLE_DEVICES=MIG-73ea7899-43fa-5c14-98d1-67eea628afd0
python examples/pytorch/gpt/multi_gpu_gpt_example.py \
            --data_type fp16 \
            --lib_path ./build_mig/lib/libth_transformer.so \
            --ckpt_path ./models/onnx-models/c-model/124m/1-gpu/ \
            --vocab_file ./gpt2/vocab.json \
            --merges_file ./gpt2/merges.txt \
            --time \
            --input_len 16 \
            --output_len 8 \
            --beam_width 8 \
            --len_penalty 1.0 \
            --max_batch_size 4 \
            --repetition_penalty 1.0 \
            --size_per_head 64 \
            --head_num 12 \
            --layer_num 12 \
            --vocab_size 50257



# Run ONNXRuntime GPT2 with beam search
pip install ort-nightly-gpu==1.15.0.dev20230329005
# ORT only support transformer 4.26
pip install transformers==v4.26.1
cd ort_benchmark
# https://github.com/microsoft/onnxruntime/blob/6a6513f9c0541e06ebede87cbf84ada0d63f1394/onnxruntime/python/tools/transformers/models/gpt2/gpt2_helper.py#L528
# Manually turn on skip_layer_norm: `optimization_options.enable_skip_layer_norm = True`, which may casuse minor output difference.
python -m onnxruntime.transformers.convert_generation -m gpt2 -p fp16 --use_gpu --output gpt2_beam_search_fp16.onnx --past_present_share_buffer --use_decoder_masked_attention

python benchmark_ort_gpt2.py




