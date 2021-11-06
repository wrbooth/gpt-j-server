"""simple api server for gpt-j"""
from flask import Flask, request, jsonify
from transformers import GPTJForCausalLM, AutoTokenizer
import torch

model = GPTJForCausalLM.from_pretrained("../gpt-j-6B", torch_dtype=torch.float16)
model = model.to(torch.device("cuda"))
tokenizer = AutoTokenizer.from_pretrained("../gpt-j-6B")

app = Flask(__name__)


@app.route('/v1/generate', methods=['POST'])
def generate():
    content = request.get_json()
    input_ids = tokenizer(content['prompt'], return_tensors="pt").input_ids
    token_len = input_ids.size(dim=1)
    reply_length = content['reply_length'] if 'reply_length' in content else 50
    max_length = content['max_length'] if 'max_length' in content else token_len + reply_length
    input_ids = input_ids.to(torch.device("cuda"))
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=max_length)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return jsonify({"generated_text": gen_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
