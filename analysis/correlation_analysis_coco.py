import matplotlib.pyplot as plt
import os, re, pickle,tqdm
import scipy.io
from collections import defaultdict
import numpy as np
import warnings, logging
warnings.filterwarnings('ignore')

logging.basicConfig(filename='/home/jinhanz/cs/xai/logs/240925_correlation_process_coco_relu_act.log', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

import pandas as pd

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def nested_dict(n):
    if n == 1:
        return defaultdict(lambda: np.ndarray(0))
    else:
        return defaultdict(lambda: nested_dict(n-1))
    
color_code = {
    "DET vs FullGradCam":       "blue",
    "DET-GrpF vs FullGradCam":  "#0072BD",
    "DET-GrpE vs FullGradCam":  "cyan",
    
    "EXP vs ODAM":              "red",
    "EXP-GrpF vs ODAM":         "orange",
    "EXP-GrpE vs ODAM":         "olive",

    "PV vs FullGradCam":        "purple",
    "PV-GrpF vs FullGradCam":   "pink",
    "PV-GrpE vs FullGradCam":   "magenta",
}

tnrfont = {'fontname':'Times New Roman'}

alpha = 0.3

"""
human attention
"""

human_attention_path = {
    "DET":'/home/jinhanz/cs/data/mscoco/human_attention/240107_DET_excluded_resized/attention_maps',
    "EXP":'/home/jinhanz/cs/data/mscoco/human_attention/231222_EXP_excluded_cleaned_resized/attention_maps',
    "PV":'/home/jinhanz/cs/data/mscoco/human_attention/231221_PV_resized/attention_maps',
}

# Attention Type, Image Idex
human_attention = {
    "DET":defaultdict(),
    "EXP":defaultdict(),
    "PV":defaultdict(),
}

for type, path_by_type in human_attention_path.items():
    for file in tqdm.tqdm(os.listdir(path_by_type)):
        img_idx = file.replace('_GSmo_21.mat','')
        mat = scipy.io.loadmat(os.path.join(path_by_type,file))
        human_attention[type][img_idx] = mat['output_map_norm']

logging.info("Finish loading human attention")

layer_name_mapping = ['model_1_act', 'model_2_cv1_act', 'model_2_cv2_act', 'model_2_m_0_cv1_act', 'model_2_m_0_cv2_act', 'model_2_cv3_act', 'model_3_act', 'model_4_cv1_act', 'model_4_cv2_act', 'model_4_m_0_cv1_act', 'model_4_m_0_cv2_act', 'model_4_m_1_cv1_act', 'model_4_m_1_cv2_act', 'model_4_cv3_act', 'model_5_act', 'model_6_cv1_act', 'model_6_cv2_act', 'model_6_m_0_cv1_act', 'model_6_m_0_cv2_act', 'model_6_m_1_cv1_act', 'model_6_m_1_cv2_act', 'model_6_m_2_cv1_act', 'model_6_m_2_cv2_act', 'model_6_cv3_act', 'model_7_act', 'model_8_cv1_act', 'model_8_cv2_act', 'model_8_m_0_cv1_act', 'model_8_m_0_cv2_act', 'model_8_cv3_act', 'model_9_cv1_act', 'model_9_cv2_act', 'model_10_act', 'model_13_cv1_act', 'model_13_cv2_act', 'model_13_m_0_cv1_act', 'model_13_m_0_cv2_act', 'model_13_cv3_act', 'model_14_act', 'model_17_cv1_act', 'model_17_cv2_act', 'model_17_m_0_cv1_act', 'model_17_m_0_cv2_act', 'model_17_cv3_act']
skip_imgs = ['book_472678',"baseball glove_515982","toothbrush_160666","potted plant_473219","bench_350607","truck_295420","toaster_232348","kite_405279","toothbrush_218439","snowboard_425906","car_227511","traffic light_453841","hair drier_239041","hair drier_178028","toaster_453302","mouse_513688","spoon_88040","scissors_340930","handbag_383842"]
expected_sample_num = 141

for is_act in ["_act"]:
    for rescale_method in ['bilinear','gaussian_sigma2','gaussian_sigma4']:
        if is_act == '' and rescale_method == 'bilinear': continue
        print(f"{is_act} {rescale_method}")
        logging.info(f"[{is_act} {rescale_method}] loading AI attention")

        """
        AI Attention
        """

        failed_imgs = {
            "FullGradCAM":defaultdict(list),
            "ODAM":defaultdict(list),
        }

        xai_saliency_path = {
            "FullGradCAM":f'/opt/jinhanz/results/debug_act/mscoco/xai_saliency_maps_yolov5s{is_act}_{rescale_method}/fullgradcamraw',
            "ODAM":f'/opt/jinhanz/results/debug_act/mscoco/xai_saliency_maps_yolov5s{is_act}_{rescale_method}/odam',
        }

        # Type, Category, Layer, Image
        xai_saliency_maps = {
            "FullGradCAM":defaultdict(defaultdict),
            "ODAM":defaultdict(defaultdict),
        }

        for type, path_by_type in xai_saliency_path.items():

            for dir in tqdm.tqdm(os.listdir(path_by_type)):
                layer_name = None
                if '.mat' in dir: continue # skip faithfulness data
                for l in layer_name_mapping:
                    if l in dir:
                        layer_name = l
                        break
                if not layer_name:
                    # print(f"NOT FOUND! {dir}")
                    continue
                # layer_num = int(re.findall(r"F\d+",dir)[-1].replace('F',''))

                for file in os.listdir(os.path.join(path_by_type,dir)):
                    if '.mat' not in file: continue
                    img_idx = file.replace('-res.mat','').replace('-res.png.mat','').replace('-res.jpg.mat','')
                    try:
                        mat = scipy.io.loadmat(os.path.join(path_by_type,dir,file))
                        # if mat['masks_ndarray'].sum()==1.5 and mat['masks_ndarray'][0,0]==1 and mat['masks_ndarray'][1,1]==0.5:
                        #     failed_imgs[type][layer_name].append(img_idx)
                        #     continue
                        # elif not np.any(mat['masks_ndarray']):
                        #     failed_imgs[type][layer_name].append(img_idx)
                        #     continue
                        if np.any(np.isnan(mat['masks_ndarray'])):
                            failed_imgs[type][layer_name].append(img_idx)
                            continue
                        xai_saliency_maps[type][layer_name][img_idx] = mat['masks_ndarray']
                    except:
                        logging.error(f"{type}\t{layer_name}\t{file}")
        logging.info("Finish loading AI attention")

        all_failed_imgs = set()
        for category, c in failed_imgs.items():
            for layer, l in c.items():
                for img in l:
                    all_failed_imgs.add(img)

        for img in skip_imgs:
            all_failed_imgs.add(img)
        logging.info(all_failed_imgs)

        # for type, t in xai_saliency_maps.items():
        #         for layer, l in t.items():
        #                 if len(l) != expected_sample_num:
        #                     logging.error(f"{type} Layer {layer} {len(l)}")

        """
        Correlation
        """

        logging.info("Start correlating")

        PCC_all = {
            "DET vs FullGradCam":defaultdict(defaultdict),
            "EXP vs ODAM":defaultdict(defaultdict),
            "PV vs FullGradCam":defaultdict(defaultdict),
        }
        RMSE_all = {
            "DET vs FullGradCam":defaultdict(defaultdict),
            "EXP vs ODAM":defaultdict(defaultdict),
            "PV vs FullGradCam":defaultdict(defaultdict),
        }

        for layer in layer_name_mapping:
            for img in human_attention['DET'].keys():
                if img in all_failed_imgs: continue

                if img in xai_saliency_maps['FullGradCAM'][layer].keys():
                    # DET vs FullGradCam
                    PCC_all['DET vs FullGradCam'][layer][img] = np.corrcoef(xai_saliency_maps['FullGradCAM'][layer][img].flatten(), human_attention['DET'][img].flatten())[0,1]
                    RMSE_all['DET vs FullGradCam'][layer][img] = RMSE(xai_saliency_maps['FullGradCAM'][layer][img].flatten(), human_attention['DET'][img].flatten())

                    # PV vs FullGradCam
                    PCC_all['PV vs FullGradCam'][layer][img] = np.corrcoef(xai_saliency_maps['FullGradCAM'][layer][img].flatten(), human_attention['PV'][img].flatten())[0,1]
                    RMSE_all['PV vs FullGradCam'][layer][img] = RMSE(xai_saliency_maps['FullGradCAM'][layer][img].flatten(), human_attention['PV'][img].flatten())

                if img in xai_saliency_maps['ODAM'][layer].keys():
                    # EXP vs ODAM
                    PCC_all['EXP vs ODAM'][layer][img] = np.corrcoef(xai_saliency_maps['ODAM'][layer][img].flatten(), human_attention['EXP'][img].flatten())[0,1]
                    RMSE_all['EXP vs ODAM'][layer][img] = RMSE(xai_saliency_maps['ODAM'][layer][img].flatten(), human_attention['EXP'][img].flatten())

        pickle.dump(PCC_all, open(f'/home/jinhanz/cs/xai/results/mscoco/240925_yolov5s_relu_act/mscoco_{is_act}{rescale_method}_PCC_all_conv.pickle','wb'))
        pickle.dump(RMSE_all, open(f'/home/jinhanz/cs/xai/results/mscoco/240925_yolov5s_relu_act/mscoco_{is_act}{rescale_method}_RMSE_all_conv.pickle','wb'))

        logging.info("Finish correlating")