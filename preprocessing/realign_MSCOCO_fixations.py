import os
import pandas as pd

imginfo_all = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/mscoco/image_info/image_info_actualarea.xlsx')

orig_path = {
    'DET':"/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/DET_excluded",
    'EXP':"/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/EXP_excluded_cleaned",
    'PV':"/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/PV",
}

output_path = {
    'DET':"/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/padding_subtracted/DET_excluded",
    'EXP':"/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/padding_subtracted/EXP_excluded_cleaned",
    'PV':"/mnt/h/OneDrive - The University Of Hong Kong/mscoco/fixation/split_by_id/padding_subtracted/PV",
}

for type in orig_path:
    if not os.path.exists(output_path[type]):
        os.makedirs(output_path[type])
    
    for file in os.listdir(orig_path[type]):

        if '.xlsx' not in file: continue

        img = file.replace('.xlsx','')
        
        fixations_img = pd.read_excel(os.path.join(orig_path[type],file))
        imginfo = imginfo_all[imginfo_all['StimuliID']==f'{img}.png']

        fixations_img['CURRENT_FIX_X'] -= imginfo['Xlo'].item()
        fixations_img['CURRENT_FIX_Y'] -= imginfo['Ylo'].item()

        fixations_img.to_excel(os.path.join(output_path[type],file),index=False)

