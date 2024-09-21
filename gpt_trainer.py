# Generate text from the fine-tuned model
input_text = "What are the implications of quantum computing?"

# Tokenize and generate
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)