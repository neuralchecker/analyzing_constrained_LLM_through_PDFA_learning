import subprocess
import re
import pandas as pd

def run():
    results = []
    number_of_executions = 5
    for i in range(number_of_executions):
        samples = [1000, 3000, 5000]
        for sample in samples:
            #this is to run outlines in another process so the cache doesnt affect the results
            p = subprocess.Popen(["python", "run_floating_point_benchmark.py", str(sample)], stdout=subprocess.PIPE)
            out = p.stdout.read()
            res = re.findall(r'\(.*?\)', str(out))
            for r in res:
                r = r.replace("(", "").replace(")", "").replace("'", "")
                r = r.split(", ")
                r[1] = int(r[1])
                r[2] = float(r[2])
                r[3] = float(r[3])
                if i != 0:
                    results.append((r[0], r[1], i, r[2], r[3]))
            print(res)


    dfresults = pd.DataFrame(results, columns=["Algorithm", "Samples", "Execution", "Generation Time", "Sample Time"])
    dfresults.to_csv('./case_studies/case_study_2_generate_floating_point/compare_time_one_digit_mask_floating_point_digits_length_n/results/' +'one_digit_mask_length_n.csv', index=False)