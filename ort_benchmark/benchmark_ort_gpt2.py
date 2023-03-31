from transformers import GPT2Tokenizer
import onnxruntime as ort
import numpy as np
import time

sess_options = ort.SessionOptions()
execution_providers = ["CUDAExecutionProvider"]
ort_session = ort.InferenceSession(
    "gpt2_beam_search_fp16.onnx", sess_options, providers=execution_providers
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
batch_size = 4
sentences = ['Apple Iphone which on is best Apple Iphone which on is best which brand'] * batch_size

inputs = tokenizer(sentences, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
max_length = 16 + 8
min_length = 2
num_beams = 8
num_return_sequences = 8
length_penalty = 1.0
repetition_penalty = 1.0

inputs = {
    "input_ids": input_ids.cpu().numpy().astype(np.int32),
    "max_length": np.array([max_length], dtype=np.int32),
    "min_length": np.array([min_length], dtype=np.int32),
    "num_beams": np.array([num_beams], dtype=np.int32),
    "num_return_sequences": np.array([num_return_sequences], dtype=np.int32),
    "length_penalty": np.array([length_penalty], dtype=np.float32),
    "repetition_penalty": np.array([repetition_penalty], dtype=np.float32),
}

for _ in range(16):
    start = time.time()
    result = ort_session.run(None, inputs)
    end = time.time()
    print(f"ort time: {(end - start) * 1000} ms")


sequences = result[0]
print("sequences", sequences)
(batch_size, num_sequences, max_length) = sequences.shape
ort_decoded_sequences = []
for i in range(batch_size):
    for j in range(num_sequences):
        decoded_sequence = tokenizer.decode(sequences[i][j][16:], skip_special_tokens=True)
        ort_decoded_sequences.append(decoded_sequence)
        print(f"batch {i} sequence {j}: {decoded_sequence}")

