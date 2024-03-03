import pandas as pd
import numpy as np
import os,re

data_path = {
    'bdd':{
        'DET Veh':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/fixation/with_duration/Veh_IdTask_60Parti_Fixation1207.xlsx',
        'DET Hum':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/fixation/with_duration/hum_id_task_60_parti_1216.xlsx',
        'EXP Veh':"/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_vehicle_whole_screen_vb_fixed_pos/veh_exp_cleaned_outliers_11_23_2023_02_18_with_duration.xlsx",
        'EXP Hum':"/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_human_whole_screen_vb_fixed_pos/hum_exp_cleaned_outliers_11_30_2023_01_11_with_duration.xlsx",
    },
    'mscoco':{
        'PV':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/with_duration_pv_fix_11_14_2023_14_45.xlsx',
        'DET':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/with_duration_detection_veh_fix_11_06_2023_17_58.xlsx',
        'EXP':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/with_duration_explanation_fix_11_14_2023_14_24.xlsx',
        'DET_excluded':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/cocluster_with_duration_det_fix_11_18_2023_12_18.xlsx',
        'EXP_excluded':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/cocluster_with_duration_exp_fix_11_18_2023_12_31.xlsx',
        'EXP_cleaned':'',
        'EXP_excluded_cleaned': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/mscoco_exp_cleaned_outliers_w_duration_12_22_2023_18_30.xlsx'
    }
}

output_path = {
    'bdd':{
        'DET Veh':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/fixation/split_by_id/Veh_Yolo_IdTask_Fixation',
        'DET Hum':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/fixation/split_by_id/Hum_Yolo_IdTask_Fixation', 
        'EXP Veh':'', # same folder as the input
        'EXP Hum':'',
    },
    'mscoco':{
        'PV': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/PV',
        'DET': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/DET',
        'EXP': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/EXP',
        ## Exclude subject with missing blocks
        'DET_excluded': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/DET_excluded',
        'EXP_excluded': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/EXP_excluded',
        'EXP_cleaned': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/EXP_cleaned',
        'EXP_excluded_cleaned': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/EXP_excluded_cleaned',
    }
}

for data in ['bdd']:
    # for condition in data_path[data]:
    for condition in ['DET Veh','DET Hum']:
        if not os.path.exists(output_path[data][condition]):
            os.makedirs(output_path[data][condition])

        df = pd.read_excel(data_path[data][condition])

        # Different label naming used by Mary and Jinhan and Alice
        if 'RECORDING_SESSION_LABEL' in df:
            df.rename(columns={'image':'img','RECORDING_SESSION_LABEL':'sessionLabel'}, inplace=True)
            df['CURRENT_FIX_Y'] -= 96 # data from Alice include padded top area
        elif 'FixD' in df:
            df.rename(columns={'FixX': 'CURRENT_FIX_X', 'FixY': 'CURRENT_FIX_Y', 'FixD':'CURRENT_FIX_DURATION','StimuliID':'img','SubjectID':'sessionLabel'}, inplace=True)
        elif 'FixDuration' in df:
            df.rename(columns={'FixX': 'CURRENT_FIX_X', 'FixY': 'CURRENT_FIX_Y', 'FixDuration':'CURRENT_FIX_DURATION','StimuliID':'img','SubjectID':'sessionLabel'}, inplace=True)
        df = df[['CURRENT_FIX_X','CURRENT_FIX_Y','CURRENT_FIX_DURATION','img','sessionLabel']]

        grouped = df.groupby('img')

        for name, group in grouped:
            name = name.replace('.png','')
            name = name.replace('.jpg','')
            if condition in ['DET Veh','DET Hum']:
                if '_' in name:
                    print('Test image included!\t' + condition + ' ' + name)
                    continue
                name = re.findall(r"\d+",name)[0]
            group.to_excel(os.path.join(output_path[data][condition],f"{name}.xlsx"),index=False)