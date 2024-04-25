# Analyzing Constrained LLM through PDFA Learning
Experiments repository for paper "Analyzing constrained LLM through PDFA-learning"

The main script for running performance experiments and generate data for case studies is './run_experiment.py'  which should be parameterized by a number we will call 'run_type'.
- run_type = 1 : performance experiment a
- run_type = 2 : performance experiment b
- run_type = 3 : case study 2 (for comparing sampling times from Outlines and an extracted PDFA)
- run_type = 4 : Case study 2 (sampling from GPT2 and PDFA, only using tokens associated to single digits)       
- run_type = 5 : Case study 2 (sampling from GPT2, PDFA and Outlines over all tokens containing digits)    

For Case study 1, refer to notebook:
'./case_studies/case_study_1_analize_man_woman/gpt2_man_woman_case_study.ipynb'
