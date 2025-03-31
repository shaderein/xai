import pandas as pd
import numpy as np
import os

exp_root_dir = "/Users/shaderein/Downloads/AntoniTask/1_ExperimentDeploy/2_Exp_Task"

# List to store DataFrames
all_data = []

for block in range(1, 5):
    for subj in range(1, 62):
        exp_dir = os.path.join(exp_root_dir, f"Exp_Task_Block{block}_deploy", "results", f"{subj:03d}ET{block}")
        exp_file_path = os.path.join(exp_dir, "RESULTS_FILE.txt")
        
        if not os.path.exists(exp_file_path):
            print(f"File not found: {exp_file_path}")
            continue
        
        exp_data = pd.read_csv(exp_file_path, delim_whitespace=True)
        
        for img in os.listdir(os.path.join(exp_dir, 'saved_images')):
            cateogry, idx, trial_num = img.replace('.png','').split('_')
            exp_data.loc[exp_data['Trial_Index_'] == int(trial_num), 'image'] = f"{cateogry}_{idx}.png"
            
        all_data.append(exp_data)
            

final_data = pd.concat(all_data, ignore_index=True)

final_data.to_csv("compiled_exp_data.csv", index=False)