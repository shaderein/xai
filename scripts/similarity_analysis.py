#%%
import matplotlib.pyplot as plt
import os, re
import scipy.io
from collections import defaultdict
import numpy as np
import pandas as pd

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#%%
human_attention_path = '/mnt/h/OneDrive - The University Of Hong Kong/bdd/results/explanation/231018_vehicle_whole_screen_vb_fixed_pos/human_saliency_map'

human_attention = defaultdict()

for file in os.listdir(human_attention_path):
    img_idx = re.findall(r'\d+_',file)[-1].replace('_','')
    mat = scipy.io.loadmat(os.path.join(human_attention_path,file))
    human_attention[img_idx] = mat['output_map_norm']

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