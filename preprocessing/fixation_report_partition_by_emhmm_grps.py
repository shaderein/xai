import pandas as pd
import numpy as np
import os, scipy, mat73

data_path = {
    'bdd':{
        'EXP Veh':"/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_vehicle_whole_screen_vb_fixed_pos/veh_exp_cleaned_outliers_11_23_2023_02_18_with_duration.xlsx",
        'EXP Hum':"/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_human_whole_screen_vb_fixed_pos/hum_exp_cleaned_outliers_11_30_2023_01_11_with_duration.xlsx",
    },
    'mscoco':{
        'PV':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/with_duration_pv_fix_11_14_2023_14_45.xlsx',
        'DET_excluded':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/cocluster_with_duration_det_fix_11_18_2023_12_18.xlsx',
        'EXP_excluded_cleaned': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/mscoco_exp_cleaned_outliers_w_duration_12_22_2023_18_30.xlsx'
    }
}

emhmm_variables = {
    'bdd':{
        'EXP Veh': '/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231225_posneg_fixed_vbhem_alpha_veh/individual_hmms.mat',
        'EXP Hum': '/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231225_posneg_fixed_vbhem_alpha_hum/individual_hmms.mat',
    },
    'mscoco':{
        'PV':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/emhmm_results/3_pv_SubjNames.mat',
        'DET_excluded':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/emhmm_results/1_it_SubjNames.mat',
        'EXP_excluded_cleaned':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/emhmm_results/2_exp_SubjNames.mat'
    }
}

emhmm_cogrp_results = {
    'bdd':{
        'EXP Veh':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231225_posneg_fixed_vbhem_alpha_veh/vbcogroup_hmms.mat',
        'EXP Hum':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231225_posneg_fixed_vbhem_alpha_hum/vbcogroup_hmms.mat'
    },
    'mscoco':{
        'PV':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/emhmm_results/3_pv_vbcogroup_hmms_1221.mat',
        'DET_excluded':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/emhmm_results/1_it_vbcogroup_hmms.mat',
        'EXP_excluded_cleaned':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/emhmm_results/2_exp_vbcogroup_hmms_1220.mat'
    }
}

output_path = {
    'bdd':{
        # Generate attention maps based on EMHMM groups
        'EXP Veh':"/mnt/h/OneDrive - The University Of Hong Kong/bdd/fixation/split_by_id/Veh_Yolo_ExpTask_Fixation",
        'EXP Hum':"/mnt/h/OneDrive - The University Of Hong Kong/bdd/fixation/split_by_id/Hum_Yolo_ExpTask_Fixation",
    },
    'mscoco':{
        'PV': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/PV',
        'DET_excluded': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/DET_excluded',
        'EXP_excluded_cleaned': '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/EXP_excluded_cleaned',
    }
}

for data in ['mscoco']:
    # for condition in ['EXP_excluded_cleaned']:
    for condition in data_path[data]:
        if not os.path.exists(f"{output_path[data][condition]}_grp1"):
            os.makedirs(f"{output_path[data][condition]}_grp1")
        if not os.path.exists(f"{output_path[data][condition]}_grp2"):
            os.makedirs(f"{output_path[data][condition]}_grp2")

        df = pd.read_excel(data_path[data][condition])

        try:
            emhmm_result = scipy.io.loadmat(emhmm_cogrp_results[data][condition])
            emhmm_grps = emhmm_result['vbco']['groups'][0][0][0] # TODO: squeeze() doesn't work?
        except: # old matlab edition used by Mary
            emhmm_result = mat73.loadmat(emhmm_cogrp_results[data][condition])
            emhmm_grps = emhmm_result['vbco']['groups'] # TODO: squeeze() doesn't work?

        emhmm_variable = scipy.io.loadmat(emhmm_variables[data][condition])
        subject_names = [emhmm_variable[k] for k in emhmm_variable.keys() if 'SubjNames' in k][0]
        subject_names = [item[0][0] for item in subject_names]

        # Different label naming used by Mary and Jinhan
        if 'FixD' in df:
            df.rename(columns={'FixX': 'CURRENT_FIX_X', 'FixY': 'CURRENT_FIX_Y', 'FixD':'CURRENT_FIX_DURATION','StimuliID':'img','SubjectID':'sessionLabel'}, inplace=True)
        elif 'FixDuration' in df:
            df.rename(columns={'FixX': 'CURRENT_FIX_X', 'FixY': 'CURRENT_FIX_Y', 'FixDuration':'CURRENT_FIX_DURATION','StimuliID':'img','SubjectID':'sessionLabel'}, inplace=True)
        df = df[['CURRENT_FIX_X','CURRENT_FIX_Y','CURRENT_FIX_DURATION','img','sessionLabel']]

        grp1_subjs = [subject_names[int(i)-1] for i in emhmm_grps[0].squeeze()]
        grp2_subjs = [subject_names[int(i)-1] for i in emhmm_grps[1].squeeze()]

        df_grp1 = df[df['sessionLabel'].isin(grp1_subjs)]
        df_grp2 = df[df['sessionLabel'].isin(grp2_subjs)]

        grouped_grp1 = df_grp1.groupby('img')
        grouped_grp2 = df_grp2.groupby('img')

        for name, group in grouped_grp1:
            group.to_excel(os.path.join(f"{output_path[data][condition]}_grp1",f"{name.replace('.png','')}.xlsx"),index=False)
        for name, group in grouped_grp2:
            group.to_excel(os.path.join(f"{output_path[data][condition]}_grp2",f"{name.replace('.png','')}.xlsx"),index=False)