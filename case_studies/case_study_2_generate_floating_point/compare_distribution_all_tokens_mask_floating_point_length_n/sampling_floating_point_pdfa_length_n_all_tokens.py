import torch
import time
import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from case_studies.gpt2.gpt2_probabilistic_model_wrapper import GPT2_probabilistic_model_wrapper
from guiding_wfas.floating_point_wfa_01_all_tokens import alphabet
from src.syncronic_model_guided_language_model import SyncronicModelGuidedLanguageModel
from guiding_wfas.floating_point_wfa_01_all_tokens import get_floating_point_wfa_01_all_tokens
from src.hypothesis_aware_sample_probabilistic_teacher import HypothesisAwareSampleProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.pdfa_operations import get_representative_sample
from pythautomata.utilities.probability_partitioner import QuantizationProbabilityPartitionerPlus


def sample():
    model_id, model, tokenizer, device = get_gpt2_model_and_tokenizer()
    wrapper = GPT2_probabilistic_model_wrapper(50, alphabet, device, model, tokenizer)
    property_model = get_floating_point_wfa_01_all_tokens(wrapper.terminal_symbol)
    synchronic_model = SyncronicModelGuidedLanguageModel(wrapper, 
                                                         property_model, 
                                                         model_name="GUIDED_GPT2", 
                                                         max_seq_length=10, 
                                                         normalize_outputs=True)
    partitioner = QuantizationProbabilityPartitionerPlus(100)
    comparator = WFAPartitionComparator(partitioner)
    max_states = 15
    max_query_length = 6
    teacher = HypothesisAwareSampleProbabilisticTeacher(synchronic_model, 
                                                        comparator = comparator, 
                                                        max_seq_length = 4, 
                                                        sample_size = 100)
    #time bound is set to 300 seconds
    learner = BoundedPDFAQuantizationNAryTreeLearner(partitioner = partitioner, 
                                                     max_states = max_states, 
                                                     max_query_length = max_query_length, 
                                                     max_seconds_run = 300, 
                                                     generate_partial_hipothesis = True, 
                                                     pre_cache_queries_for_building_hipothesis = True,  
                                                     check_probabilistic_hipothesis = True, 
                                                     mean_distribution_for_partial_hipothesis = True, 
                                                     omit_zero_transitions = True)

    

    learning_result = learner.learn(teacher, verbose=True)
    pdfa = learning_result.model

    floating_points = []
  
    sample_time = 0
    start_time = time.time()
    for i in range(10000):
        number = get_representative_sample(pdfa, sample_size = 1, max_length = 3)
        number_string = str(number)
        
        result = number_string.replace('[', '').replace(']', '').replace(',', '')

        floating_points.append(result)

    sample_time = time.time() - start_time

    df = pd.DataFrame(floating_points, columns=["floating-point"])
    df.to_csv("./case_studies/case_study_2_generate_floating_point/compare_distribution_all_tokens_mask_floating_point_length_n/results/pdfa_10k_all_tokens_length_n.csv", index=False)


    
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