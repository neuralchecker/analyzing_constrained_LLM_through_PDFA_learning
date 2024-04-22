from case_studies.case_study_2_generate_floating_point.compare_time_one_digit_mask_floating_point_digits_length_n.sampling_floating_point_digits_length_n import benchmark_algorithms
import sys
import argparse


if __name__ == "__main__":
    if len(sys.argv) > 1:
        value = int(sys.argv[1])
    else:
        raise ValueError("Please provide a value as a command line argument.")
    benchmark_algorithms(sample_size=value)