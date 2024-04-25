import torch
import time
import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import outlines
from outlines.models.transformers import Transformers

def sample():
    model_id, model, tokenizer, device = get_gpt2_model_and_tokenizer()
    outlinesModel = Transformers(model, tokenizer)
    bos_outlines = tokenizer.decode(tokenizer.bos_token_id)
    prompt = bos_outlines
    outlinesGenerator = outlines.generate.regex(outlinesModel, "\.[0-9]+")

    floating_points = []
    for i in range(10000):
        _ = outlinesGenerator(prompt)
        floating_points.append(_)

    
    df = pd.DataFrame(floating_points, columns=["floating-point"])
    df.to_csv("./case_studies/case_study_2_generate_floating_point/compare_distribution_all_tokens_mask_floating_point_length_n/results/outlines_10k_length_n.csv", index=False)

def get_gpt2_model_and_tokenizer():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, add_prefix_space=False)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                return_dict_in_generate=True,
                                                pad_token_id=tokenizer.eos_token_id).to(device)
                                                
    return model_id, model, tokenizer, device


if __name__ == "__main__":
    sample()