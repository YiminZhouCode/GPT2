import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.model_training import train_model, incremental_training

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load data
processed_data_dir = "data/processed_data/"
original_data_dir = "data/raw_data/"

encoded_inputs = [tokenizer(open(os.path.join(processed_data_dir, f)).read(), return_tensors='pt') for f in os.listdir(processed_data_dir)]
original_inputs = [tokenizer(open(os.path.join(original_data_dir, f)).read(), return_tensors='pt') for f in os.listdir(original_data_dir)]

# Initial training
train_model(encoded_inputs, model, tokenizer, epochs=3)

# Incremental training
incremental_training(encoded_inputs, original_inputs, model, tokenizer, epochs=3)

# Save the fine-tuned model
model.save_pretrained("./models/fine_tuned_gpt2")
tokenizer.save_pretrained("./models/fine_tuned_gpt2")
