import spacy

nlp = spacy.load("en_core_web_sm")

def syntactic_compression(text):
    doc = nlp(text)
    compressed_tokens = []
    for token in doc:
        if token.dep_ in {"nsubj", "ROOT", "dobj"}:
            compressed_tokens.append(token.text)
    return " ".join(compressed_tokens)

def preprocess_data(input_dir, output_dir):
    import os
    for filename in os.listdir(input_dir):
        with open(os.path.join(input_dir, filename), 'r') as f:
            text = f.read()
        compressed_text = syntactic_compression(text)
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(compressed_text)
