import pandas as pd
import numpy as np

# EXP Veh
# data_path = "/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_vehicle_whole_screen_vb_fixed_pos/veh_exp_cleaned_outliers_11_23_2023_02_18_with_duration.xlsx"

# EXP Hum
data_path = "/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_human_whole_screen_vb_fixed_pos/hum_exp_cleaned_outliers_11_30_2023_01_11_with_duration.xlsx"

# DET Veh
data_path = "/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_human_whole_screen_vb_fixed_pos/hum_exp_cleaned_outliers_11_30_2023_01_11_with_duration.xlsx"

# DET Hum
data_path = "/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_human_whole_screen_vb_fixed_pos/hum_exp_cleaned_outliers_11_30_2023_01_11_with_duration.xlsx"


df = pd.read_excel(data_path)
df.rename(columns={'FixX': 'CURRENT_FIX_X', 'FixY': 'CURRENT_FIX_Y', 'FixD':'CURRENT_FIX_DURATION','StimuliID':'img','SubjectID':'sessionLabel'}, inplace=True)
df = df[['CURRENT_FIX_X','CURRENT_FIX_Y','CURRENT_FIX_DURATION','img','sessionLabel']]

grouped = df.groupby('img')

for name, group in grouped:
    group.to_excel(f"{name.replace('.jpg','')}.xlsx",index=False)