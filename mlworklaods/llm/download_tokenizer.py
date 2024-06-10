from transformers import AutoTokenizer

# Specify the tokenizer name
tokenizer_name = "EleutherAI/pythia-14m"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# Access the EOS token
eos_token = tokenizer.eos_token
# Example text
text = "Hello, Hugging Face!"

# Encode the text
encoded_input = tokenizer(text, return_tensors='pt')

# Print the encoded input
print("Encoded Input:", encoded_input)
