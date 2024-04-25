import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

def calculate_probs(prompt, eos, model, tokenizer):
    str_seq = [tokenizer.tokenize(x) for x in prompt]
    str_seq = [item for tokens in str_seq for item in tokens]
    prompt_ids = tokenizer.convert_tokens_to_ids(str_seq)        
    input_ids = torch.tensor(prompt_ids).reshape(1, -1)  
    with torch.no_grad():
            output = model(input_ids)
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)[0]
 
 
    numbers = ["0", "1", "2","3","4","5","6","7","8","9"]
    indexes = [tokenizer.encode(number) for number in numbers]
    if eos:
        indexes.append([tokenizer.eos_token_id])
    word_probs = {}
    for i in indexes:
        word_prob = probs[i]
        word_probs[tokenizer.decode(i).replace(" ","")] = word_prob.item()
    normalized_word_probs = {}
    total = sum(word_probs.values())
    for word in word_probs:
        normalized_word_probs[word] = word_probs[word] / total
    return normalized_word_probs

def sample():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                return_dict_in_generate=True,
                                                pad_token_id=tokenizer.eos_token_id).to(device)

    results = []
    for i in range(10000):
        next_token = ""
        prompt = [tokenizer.decode(tokenizer.bos_token_id) , "."]
        min_digits = 1
        max_digits = np.inf
        while next_token != tokenizer.decode(tokenizer.eos_token_id):            
            if len(prompt) > min_digits+1:
                normalized_word_probs = calculate_probs(prompt, True, model, tokenizer)
            else:
                normalized_word_probs = calculate_probs(prompt, False, model, tokenizer)      
        
            next_token = np.random.choice(a=list(normalized_word_probs), p=list(normalized_word_probs.values()))
            if next_token != tokenizer.decode(tokenizer.eos_token_id):
                prompt.append(next_token)                        
            if len(prompt)>=max_digits:
                next_token = tokenizer.decode(tokenizer.eos_token_id)
        results.append(''.join(prompt[1:]))
        
    df = pd.DataFrame(results, columns=["floating-point"])
    df.to_csv("./case_studies/case_study_2_generate_floating_point/compare_distribution_one_digit_mask_floating_point_length_n/results/llm_10k_length_n.csv", index=False) 


