from airllm import AutoModel

# Initialize model (automatically detects model type)
model = AutoModel.from_pretrained("openai/gpt-oss-120b",
                                 compression='4bit')  # Optional compression

# Tokenize input
input_text = ['What is the capital of United States?']
input_tokens = model.tokenizer(input_text,
                               return_tensors="pt", 
                               return_attention_mask=False, 
                               truncation=True, 
                               max_length=128)

# Generate
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
   
    se_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])
print("Output:", output)