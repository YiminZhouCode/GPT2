from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch

def train_model(encoded_inputs, model, tokenizer, epochs=3, lr=5e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for encoded_input in encoded_inputs:
            outputs = model(**encoded_input, labels=encoded_input['input_ids'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")

def incremental_training(encoded_inputs, original_inputs, model, tokenizer, epochs=3, lr=5e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for encoded_input, original_input in zip(encoded_inputs, original_inputs):
            combined_input_ids = torch.cat((encoded_input['input_ids'], original_input['input_ids']), dim=1)
            combined_attention_mask = torch.cat((encoded_input['attention_mask'], original_input['attention_mask']), dim=1)
            outputs = model(input_ids=combined_input_ids, attention_mask=combined_attention_mask, labels=combined_input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1} Incremental Loss: {loss.item()}")
