# # models/model_loader.py
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# def load_llama_model(model_name="TheBloke/Nous-Hermes-Llama2-GPTQ"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
#     return model, tokenizer

import torch
print(torch.cuda.is_available())
