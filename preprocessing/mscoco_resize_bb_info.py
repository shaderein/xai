import os, cv2
import pandas as pd

# Provided by Chenyang, with original coco sizes
bb_selections = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/mscoco/other/for_eyegaze_GT_infos.xlsx')

original_imgs_path = '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/raw'

for img_path in os.listdir(original_imgs_path):
    original_img = cv2.imread(os.path.join(original_imgs_path, img_path))
    idx = bb_selections.index[bb_selections['img']==img_path.split('/')[-1].replace('.jpg','.png')].to_list()[0]
    bb_selections.at[idx,'y1'] = bb_selections.at[idx,'y1'] / original_img.shape[0]
    bb_selections.at[idx,'x1'] = bb_selections.at[idx,'x1'] / original_img.shape[1]
    bb_selections.at[idx,'y2'] = bb_selections.at[idx,'y2'] / original_img.shape[0]
    bb_selections.at[idx,'x2'] = bb_selections.at[idx,'x2'] / original_img.shape[1]

bb_selections.to_excel('/mnt/h/OneDrive - The University Of Hong Kong/mscoco/other/for_eyegaze_GT_infos_size_ratio.xlsx')