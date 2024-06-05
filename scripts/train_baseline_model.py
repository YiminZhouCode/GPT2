import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch
from src.config import Config
from src.utils import save_to_file

def load_data(data_dir):
    files = os.listdir(data_dir)
    texts = []
    for file in files:
        with open(os.path.join(data_dir, file), 'r') as f:
            texts.append(f.read())
    return texts

def encode_texts(texts, tokenizer):
    encoded_inputs = [tokenizer(text, return_tensors='pt') for text in texts]
    return encoded_inputs

def train_model(encoded_inputs, model, epochs, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        for encoded_input in encoded_inputs:
            outputs = model(**encoded_input, labels=encoded_input['input_ids'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")

if __name__ == "__main__":
    # Load and encode raw data
    raw_texts = load_data(Config.RAW_DATA_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encoded_inputs = encode_texts(raw_texts, tokenizer)

    # Load pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Train the model
    train_model(encoded_inputs, model, Config.EPOCHS, Config.LEARNING_RATE)

    # Save the fine-tuned model
    model_dir = Config.BASELINE_MODEL_DIR
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"Model saved to {model_dir}")
