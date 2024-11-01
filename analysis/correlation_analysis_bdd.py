import matplotlib.pyplot as plt
import os, re, pickle,tqdm, torch
import scipy.io
from collections import defaultdict
import numpy as np
import warnings, logging
warnings.filterwarnings('ignore')

logging.basicConfig(filename='/home/jinhanz/cs/xai/logs/241009_correlation_process_bdd.log', 
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
}

tnrfont = {'fontname':'Times New Roman'}

alpha = 0.3

for object in ['vehicle']:

    """
    human attention
    """
    if object == 'vehicle':
        skip_imgs = ["178", "54", "452", "478", "629", "758", "856",'1007', '1028', '1041', '1065', '1100', '1149', '1236', '1258', '1272', '1331', '1356', '210', '222', '3', '390', '431', '485', '505', '52', '559', '585', '634', '648', '670', '715', '784', '797', '803', '833', '848', '867', '899', '914', '940', '980', '993','1121', '1127', '1170', '1365', '321', '425', '542', '610', '896', '902', '953', '967']
        human_attention_path = {
                "DET":'/home/jinhanz/cs/data/bdd/human_attention/240107 Veh DET/whole_image',
                "EXP":'/home/jinhanz/cs/data/bdd/human_attention/240918 Veh EXP/human_saliency_map',
            }
    elif object == 'human':
        skip_imgs = ['1022', '1041', '1053', '1063', '1066', '1097', '11', '1141', '1142', '1154', '1227', '1228', '1273', '1293', '1302', '1313', '1346', '1359', '1398', '1420', '1430', '1475', '1506', '152', '1538', '1553', '1624', '1663', '1664', '1746', '1770', '1788', '1803', '1805', '1817', '1852', '186', '1863', '1893', '19', '1917', '1954', '2008', '2040', '2087', '2092', '2108', '2121', '2128', '2141', '2161', '2186', '2203', '2219', '2226', '2262', '2270', '2271', '2279', '231', '2312', '2327', '2334', '2457', '250', '286', '348', '388', '391', '415', '422', '425', '452', '47', '608', '670', '683', '748', '757', '805', '808', '829', '845', '85', '875', '897', '900', '928', '962', '97', '997']
        human_attention_path = {
            "DET":'/home/jinhanz/cs/data/bdd/human_attention/240107 Hum DET/whole_image',
            "EXP":'/home/jinhanz/cs/data/bdd/human_attention/240918 Hum EXP/human_saliency_map',
        }


    # Attention Type, Image Idex
    human_attention = {
        "DET":defaultdict(),
        "EXP":defaultdict(),
        "PV":defaultdict(),
    }

    for type, path_by_type in human_attention_path.items():
        for file in tqdm.tqdm(os.listdir(path_by_type)):
            img_idx = file.replace('_GSmo_30.mat','')
            mat = scipy.io.loadmat(os.path.join(path_by_type,file))
            human_attention[type][img_idx] = mat['output_map_norm']

    logging.info(f"[Category - {object}] Finish loading human attention")

    layer_name_mapping = ['model_1_act', 'model_2_cv1_act', 'model_2_cv2_act', 'model_2_m_0_cv1_act', 'model_2_m_0_cv2_act', 'model_2_cv3_act', 'model_3_act', 'model_4_cv1_act', 'model_4_cv2_act', 'model_4_m_0_cv1_act', 'model_4_m_0_cv2_act', 'model_4_m_1_cv1_act', 'model_4_m_1_cv2_act', 'model_4_cv3_act', 'model_5_act', 'model_6_cv1_act', 'model_6_cv2_act', 'model_6_m_0_cv1_act', 'model_6_m_0_cv2_act', 'model_6_m_1_cv1_act', 'model_6_m_1_cv2_act', 'model_6_m_2_cv1_act', 'model_6_m_2_cv2_act', 'model_6_cv3_act', 'model_7_act', 'model_8_cv1_act', 'model_8_cv2_act', 'model_8_m_0_cv1_act', 'model_8_m_0_cv2_act', 'model_8_cv3_act', 'model_9_cv1_act', 'model_9_cv2_act', 'model_10_act', 'model_13_cv1_act', 'model_13_cv2_act', 'model_13_m_0_cv1_act', 'model_13_m_0_cv2_act', 'model_13_cv3_act', 'model_14_act', 'model_17_cv1_act', 'model_17_cv2_act', 'model_17_m_0_cv1_act', 'model_17_m_0_cv2_act', 'model_17_cv3_act']

    for is_act in ['']:
        for rescale_method in ['bilinear']:
            print(f"{object} {rescale_method}")
            logging.info(f"[{object} {rescale_method}] loading AI attention")

            """
            AI Attention
            """

            failed_imgs = {
                "FullGradCAM":defaultdict(list),
                "ODAM":defaultdict(list),
            }

            xai_saliency_path = {
                "FullGradCAM":f'/opt/jinhanz/results/optimize_faithfulness/bdd/activation_maps_yolov5s/fullgradcamraw_{object}',
                "ODAM":f'/opt/jinhanz/results/optimize_faithfulness/bdd/activation_maps_yolov5s/odam_{object}',
            }

            # Type, Category, Layer, Image
            xai_saliency_maps = {
                "FullGradCAM":defaultdict(defaultdict),
                "ODAM":defaultdict(defaultdict),
            }

            for type, path_by_type in xai_saliency_path.items():

                for dir in tqdm.tqdm(os.listdir(path_by_type)):
                    layer_name = None
                    if '.pickle' in dir: continue # skip faithfulness data
                    for l in layer_name_mapping:
                        if l in dir:
                            layer_name = l
                            break
                    if not layer_name:
                        # print(f"NOT FOUND! {dir}")
                        continue
                    # layer_num = int(re.findall(r"F\d+",dir)[-1].replace('F',''))

                    for file in os.listdir(os.path.join(path_by_type,dir)):
                        if '.pth' not in file: continue
                        img_idx = file.replace('-res.pth','').replace('-res.png.pth','').replace('-res.jpg.pth','')
                        try:
                            mat = torch.load(os.path.join(path_by_type,dir,file))
                            if mat['masks_ndarray'].sum()==1.5 and mat['masks_ndarray'][0,0]==1 and mat['masks_ndarray'][1,1]==0.5:
                                failed_imgs[type][layer_name].append(img_idx)
                                continue
                            elif not np.any(mat['masks_ndarray']):
                                failed_imgs[type][layer_name].append(img_idx)
                                continue
                            elif np.any(np.isnan(mat['masks_ndarray'])):
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
            }
            RMSE_all = {
                "DET vs FullGradCam":defaultdict(defaultdict),
                "EXP vs ODAM":defaultdict(defaultdict),
            }

            for layer in layer_name_mapping:
                for img in human_attention['DET'].keys():
                    if img in all_failed_imgs: continue

                    if img in xai_saliency_maps['FullGradCAM'][layer].keys():
                        # DET vs FullGradCam
                        PCC_all['DET vs FullGradCam'][layer][img] = np.corrcoef(xai_saliency_maps['FullGradCAM'][layer][img].flatten(), human_attention['DET'][img].flatten())[0,1]
                        RMSE_all['DET vs FullGradCam'][layer][img] = RMSE(xai_saliency_maps['FullGradCAM'][layer][img].flatten(), human_attention['DET'][img].flatten())

                    if img in xai_saliency_maps['ODAM'][layer].keys():
                        # EXP vs ODAM
                        PCC_all['EXP vs ODAM'][layer][img] = np.corrcoef(xai_saliency_maps['ODAM'][layer][img].flatten(), human_attention['EXP'][img].flatten())[0,1]
                        RMSE_all['EXP vs ODAM'][layer][img] = RMSE(xai_saliency_maps['ODAM'][layer][img].flatten(), human_attention['EXP'][img].flatten())

            save_dir = f'/home/jinhanz/cs/xai/results/bdd/241024_optimize_faithfulness_activation_maps_yolov5s'
            os.makedirs(save_dir,exist_ok=True)
            pickle.dump(PCC_all, open(f'{save_dir}/{object}_PCC_all_conv.pickle','wb'))
            pickle.dump(RMSE_all, open(f'{save_dir}/{object}_RMSE_all_conv.pickle','wb'))

            logging.info("Finish correlating")