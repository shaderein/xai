import cv2, re, os
import pandas as pd

"""
Crop the experiment images based on imginfo_actualarea to revert back to the resized/unpadded images
TODO: should ask Chenyang on the original ones
"""

input_path = {
    'DET':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/padded/DET',
    'EXP':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/padded/EXP'
}

output_path = {
    'DET':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/DET',
    'EXP':'/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/EXP'
}

imginfo = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/mscoco/image_info/image_info_actualarea.xlsx',index_col=0)

for condition, imgs_path in input_path.items():
    for img_path in os.listdir(imgs_path):
        img_name = img_path.replace('.png','')

        if '.png' not in img_path or img_path not in imginfo.index: continue
        info = imginfo.loc[img_path]

        img = cv2.imread(os.path.join(imgs_path,img_path))

        img = img[info['Ylo']:info['Yhi'],info['Xlo']:info['Xhi']]
        cv2.imwrite(os.path.join(output_path[condition],img_path),img)