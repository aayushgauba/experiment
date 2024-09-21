import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "gpt2"  # You can use any pre-trained model, such as GPT-2
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Example text to compute perplexity
text = "Explain the theory of relativity in simple terms."

# Tokenize the input text
input_ids = tokenizer.encode(text, return_tensors="pt")

# Disable gradient calculations for evaluation
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss

# Compute perplexity
perplexity = torch.exp(loss)
print(f"Perplexity: {perplexity.item()}")