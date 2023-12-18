#%%
import matplotlib.pyplot as plt
import os, re
import scipy.io
from collections import defaultdict
import numpy as np
import pandas as pd

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def nested_dict(n, type=object):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

#%%
human_attention_path = {
    "DET":{
        'vehicle': '/mnt/h/OneDrive - The University Of Hong Kong/bdd/attention_maps/231206 Veh DET/whole_image',
        'human':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/attention_maps/231206 Hum DET/whole_image',
    },
    "DET-Cropped":{
        'vehicle': '/mnt/h/OneDrive - The University Of Hong Kong/bdd/attention_maps/231206 Veh DET/cropped',
        'human':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/attention_maps/231206 Hum DET/cropped',
    },
    "EXP":{
        'vehicle':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_vehicle_whole_screen_vb_fixed_pos/human_saliency_map',
        'human':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_human_whole_screen_vb_fixed_pos/human_saliency_map',
    }
}

# Attention Type, Category, Image Idex
human_attention = nested_dict(3)

for type, path_by_type in human_attention_path.items():
    for category, path in path_by_type.items():
        for file in os.listdir(path):
            img_idx = re.findall(r'\d+',file)[0]
            mat = scipy.io.loadmat(os.path.join(path,file))
            human_attention[type][category][img_idx] = mat['output_map_norm']

#%%
xai_saliency_path = '/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/multi_layer_analysis/odam_test_results'

xai_saliency_maps = defaultdict(defaultdict)
PCC_all = defaultdict(defaultdict)
RMSE_all = defaultdict(defaultdict)

for dir in os.listdir(xai_saliency_path):
    layer_num = int(re.findall(r"F\d+",dir)[-1].replace('F',''))

    for file in os.listdir(os.path.join(xai_saliency_path,dir)):
        if '.mat' not in file: continue
        img_idx = re.findall(r'\d+-',file)[-1].replace('-','')
        mat = scipy.io.loadmat(os.path.join(xai_saliency_path,dir,file))
        xai_saliency_maps[layer_num][img_idx] = mat['masks_ndarray']

        PCC_all[layer_num][img_idx] = np.corrcoef(xai_saliency_maps[layer_num][img_idx].flatten(), human_attention[img_idx].flatten())[0,1]
        RMSE_all[layer_num][img_idx] = RMSE(xai_saliency_maps[layer_num][img_idx].flatten(), human_attention[img_idx].flatten())

#%%
PCC_layer_mean = pd.DataFrame.from_dict(PCC_all).mean(axis=0)
PCC_sorted = PCC_layer_mean.sort_index()
RMSE_layer_mean = pd.DataFrame.from_dict(RMSE_all).mean(axis=0)
RMSE_sorted = RMSE_layer_mean.sort_index()

#%%
plt.figure()
plt.plot(PCC_sorted.index, PCC_sorted.values, marker='o',fillstyle='none')
plt.xticks(PCC_sorted.index)
plt.ylim((0.2,0.8))
plt.grid()
plt.show()

#%%
plt.figure()
plt.plot(RMSE_sorted.index, RMSE_sorted.values, marker='o',fillstyle='none')
plt.xticks(RMSE_sorted.index)
plt.ylim((0,0.3))
plt.grid()
plt.show()