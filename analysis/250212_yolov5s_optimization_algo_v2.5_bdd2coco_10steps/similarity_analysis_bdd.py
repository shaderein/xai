import matplotlib.pyplot as plt
import os, re, pickle,tqdm, torch, sys
import scipy.io
from collections import defaultdict
import numpy as np
import warnings, logging
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

logging.basicConfig(filename='/home/jinhanz/cs/xai/logs/250212_yolov5s_correlation_process_bdd2coco.log', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

import pandas as pd

"""
Analysis
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity

# Define similarity functions
def normalize_map(att_map):
    return att_map / np.sum(att_map)

def compute_jsd(map1, map2):
    map1 = normalize_map(map1).flatten()
    map2 = normalize_map(map2).flatten()
    return jensenshannon(map1, map2)**2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from geomloss import SamplesLoss
import torchvision, torch

def get_grids(s_x, s_y, mask_shape):
    H, W = mask_shape
    shifts_x = torch.arange(0, W*s_x, step=s_x)[:W]
    shifts_y = torch.arange(0, H*s_y, step=s_y)[:H]
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    grids = torch.stack((shift_x, shift_y), dim=1).cuda()
    grids[:,0] += s_x / 2
    grids[:,1] += s_y / 2
    return grids

def coor_dist(X, Y):
    """
    X: x map's coordinates (M,2)
    y: y map's coordinates (N,2)
    return distance (M,N)
    """
    x_col = X.unsqueeze(1)
    y_lin = Y.unsqueeze(0)
    C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
    C = torch.sqrt(C)
    return C

def compute_otd(eyegaze, heatmap):
    """
    Compute Optimal Transport Distance (OTD) between eyegaze and heatmap.

    Arguments:
    - eyegaze: Tensor of the eyegaze heatmap (2D).
    - heatmap: Tensor of the saliency heatmap (2D).
    - bbox_normalizer: Normalizer based on bounding box size (optional).

    Returns:
    - OTD (float): Optimal transport distance.
    """
    eyegaze = torch.tensor(eyegaze).cuda()
    heatmap = torch.tensor(heatmap).cuda()

    H,W = eyegaze.shape
    resize = torchvision.transforms.Resize([int(H/8),int(W/8)])
    eyegaze = resize(eyegaze.unsqueeze(0))
    heatmap = resize(heatmap.unsqueeze(0))
    _,h,w = eyegaze.shape

    map_grids = get_grids(H/h, W/w, (h,w))
    dist = coor_dist(map_grids, map_grids)
    dist = dist.unsqueeze(0)
    cost = [dist, dist.clone(), dist.clone(), dist.clone()]
                
    # Flatten the heatmaps and normalize to probability distributions
    alpha = eyegaze.flatten() / eyegaze.sum()  # Probability distribution
    beta = heatmap.flatten() / heatmap.sum()  # Probability distribution
    
    # Define Sinkhorn-based SamplesLoss
    sample_loss = SamplesLoss(
        "sinkhorn", 
        p=1, 
        blur=0.7, 
        scaling=0.5, 
        debias=True, 
        backend="tensorized"
    )
    
    # Compute the optimal transport distance
    cost_loss, _, _ = sample_loss(alpha, alpha.new_ones(h * w, 1), beta, beta.new_ones(h * w, 1), cost)
    
    return cost_loss.item()

def compute_correlation(map1, map2):
    return np.corrcoef(map1.flatten(), map2.flatten())[0, 1]

def compute_RMSE(map1, map2):
    return np.sqrt(((map1.flatten() - map2.flatten()) ** 2).mean())

def compute_cosine_similarity(map1, map2):
    return cosine_similarity(map1.reshape(1, -1), map2.reshape(1, -1))[0, 0]

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

for object in ['human']: #,

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
    for rescale_method in ['optimize_faithfulness_finer_v2.5_10steps']: # ,
        for is_act in ["xai_saliency"]: #,
            
            print(f"{object} {rescale_method} {is_act}")
            logging.info(f"[{object} {rescale_method} {is_act}] loading AI attention")

            """
            AI Attention
            """

            failed_imgs = {
                "FullGradCAM":defaultdict(list),
                "ODAM":defaultdict(list),
            }

            xai_saliency_path = {
                "FullGradCAM":f'/opt/jinhanz/results/yolov5s/{rescale_method}/bdd2coco/{is_act}_maps_yolov5s/fullgradcamraw_{object}',
                "ODAM":f'/opt/jinhanz/results/yolov5s/{rescale_method}/bdd2coco/{is_act}_maps_yolov5s/odam_{object}',
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
                        img_idx = file.replace('-res.pth','').replace('-res.pth','').replace('-res.png.pth','').replace('-res.jpg.pth','')
                        mat = torch.load(os.path.join(path_by_type,dir,file))
                        if mat['masks_ndarray'].sum()==1.5 and mat['masks_ndarray'][0,0]==1 and mat['masks_ndarray'][1,1]==0.5:
                            failed_imgs[type][layer_name].append(img_idx)
                            continue
                        elif not np.any(mat['masks_ndarray']):
                            failed_imgs[type][layer_name].append(img_idx)
                            continue
                        if np.any(np.isnan(mat['masks_ndarray'])):
                            failed_imgs[type][layer_name].append(img_idx)
                            continue
                        xai_saliency_maps[type][layer_name][img_idx] = mat['masks_ndarray']
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

            similarity_methods = {
                # "RMSE": compute_RMSE,
                "OTD": compute_otd,
                "JSD": compute_jsd,
                "Correlation": compute_correlation,
                "CosineSim": compute_cosine_similarity,
            }

            for method, func in similarity_methods.items():

                save_dir = f'/home/jinhanz/cs/xai/results/yolov5s/bdd2coco/250212_yolov5s_bdd2coco_{rescale_method}_{is_act}_maps'
                save_path = f'{save_dir}/{object}_{method}_all_conv.pickle'

                # if os.path.exists(save_path): continue

                logging.info(f"Start {method}")

                results_all = {
                    "DET vs FullGradCam":defaultdict(defaultdict),
                    "EXP vs ODAM":defaultdict(defaultdict),
                }
            
                for layer in tqdm.tqdm(layer_name_mapping):
                    for img in human_attention['DET'].keys():
                        if img in all_failed_imgs: continue

                        if img in xai_saliency_maps['FullGradCAM'][layer].keys():
                            # DET vs FullGradCam
                            results_all['DET vs FullGradCam'][layer][img] = func(xai_saliency_maps['FullGradCAM'][layer][img], human_attention['DET'][img])

                        if img in xai_saliency_maps['ODAM'][layer].keys():
                            # EXP vs ODAM
                            results_all['EXP vs ODAM'][layer][img] = func(xai_saliency_maps['ODAM'][layer][img], human_attention['EXP'][img])

                os.makedirs(save_dir,exist_ok=True)
                pickle.dump(results_all, open(save_path,'wb'))

                logging.info(f"Finish {method}")