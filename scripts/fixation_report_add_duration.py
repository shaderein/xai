import pandas as pd
import numpy as np

# Not used...

data_path = "/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_vehicle_whole_screen_vb_fixed_pos/veh_exp_cleaned_outliers_10_23_2023_23_15_pos_fixed.xlsx"


df = pd.read_excel(data_path)


"""
# Extract fixation duration data from the raw report and 
#   insert into the cleaned data as duration is needed 
#   when generating human attention maps
raw_data_path = "/mnt/h/OneDrive - The University Of Hong Kong/bdd/fixation/with_duration/veh_exp_screen_11_22_2023_23_25.xlsx"
df_raw = pd.read_excel(raw_data_path)

df_raw_format = df_raw
df_raw_format['FixY'] = df_raw_format['FixY'] - 30

# Merge dataframes on common columns
merged_df = pd.merge(df_raw_format, df, on=['SubjectID','StimuliID','FixX','FixY'], how='inner')

# Extract the indices of matching rows in df_raw_format
indices = df_raw_format.loc[df_raw_format.set_index(['SubjectID','StimuliID','FixX','FixY']).index.isin(df.set_index(['SubjectID', 'TrialID','StimuliID','FixX','FixY']).index)].index

filtered = df_raw_format.loc[indices]
"""

df.rename(columns={'FixX': 'CURRENT_FIX_X', 'FixY': 'CURRENT_FIX_Y','StimuliID':'img','SubjectID':'sessionLabel'}, inplace=True)

grouped = df.groupby('')