import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def evaluate_model(model_dir, sample_text):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    inputs = tokenizer.encode(sample_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

    print("Generated Text:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    model_dir = "../models/fine_tuned_gpt2/"
    sample_text = "The quick brown fox"
    evaluate_model(model_dir, sample_text)
