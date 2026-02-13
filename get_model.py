from transformers import AutoModel, AutoTokenizer

# Download and cache a model
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"  # or whatever model you need
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

