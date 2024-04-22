from performance_experiments.compare_on_random_pdfa_varying_zero_probabilities import performance_experiment_a_compare_on_random_pdfa
from performance_experiments.compare_on_random_pdfa_varying_nominal_size import performance_experiment_b_compare_on_random_pdfa
import run_floating_point_benchmark_in_subprocesses
from case_studies.case_study_2_generate_floating_point.compare_distribution_one_digit_mask_floating_point_length_n import sampling_floating_point_gpt2_length_n, sampling_floating_point_pdfa_length_n
from case_studies.case_study_2_generate_floating_point.compare_distribution_all_tokens_mask_floating_point_length_n import sampling_floating_point_gpt2_length_n_all_tokens, sampling_floating_point_outlines_length_n_all_tokens, sampling_floating_point_pdfa_length_n_all_tokens
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

import sys


if __name__ == '__main__':
    args = sys.argv   
    run_type = int(args[1]) 
    if run_type == 1:
        print("Running performance experiment a:")
        performance_experiment_a_compare_on_random_pdfa.run()
    if run_type == 2:
        print("Running performance experiment b:")
        performance_experiment_b_compare_on_random_pdfa.run()
    elif run_type == 3:
        print("Running experiment for 'Case study 2' comparing algorithm times:")
        run_floating_point_benchmark_in_subprocesses.run()
    elif run_type == 4:
        print("Running experiment for 'Case study 2' sampling from GPT2 and PDFA:")        
        sampling_floating_point_gpt2_length_n.sample()
        sampling_floating_point_pdfa_length_n.sample()
    elif run_type == 5:
        print("Running experiment for 'Case study 2' sampling from GPT2, PDFA and Outlines over all tokens containing digits:")
        sampling_floating_point_gpt2_length_n_all_tokens.sample()
        sampling_floating_point_pdfa_length_n_all_tokens.sample()
        sampling_floating_point_outlines_length_n_all_tokens.sample()