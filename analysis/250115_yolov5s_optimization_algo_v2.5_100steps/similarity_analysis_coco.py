import matplotlib.pyplot as plt
import os, re, pickle,tqdm, torch, sys
import scipy.io
from collections import defaultdict
import numpy as np
import warnings, logging
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
logging.basicConfig(filename='/home/jinhanz/cs/xai/logs/250115_yolov5s_correlation_process_coco_optimize_faithfulness_.log', 
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

for rescale_method in ['bilinear']: # 'optimize_faithfulness_finer_v2.5',
    for is_act in ["xai_saliency","activation"]:
        # print(f"{is_act} {rescale_method}")
        # logging.info(f"[{is_act} {rescale_method}] loading AI attention")

        """
        AI Attention
        """

        failed_imgs = {
            "FullGradCAM":defaultdict(list),
            "ODAM":defaultdict(list),
        }

        xai_saliency_path = {
            "FullGradCAM":f'/opt/jinhanz/results/yolov5s/{rescale_method}/mscoco/{is_act}_maps_yolov5s/fullgradcamraw',
            "ODAM":f'/opt/jinhanz/results/yolov5s/{rescale_method}/mscoco/{is_act}_maps_yolov5s/odam',
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
            "JSD": compute_jsd,
            "OTD": compute_otd,
            "Correlation": compute_correlation,
            "CosineSim": compute_cosine_similarity,
        }

        for method, func in similarity_methods.items():

            save_dir = f'/home/jinhanz/cs/xai/results/mscoco/250115_{rescale_method}_{is_act}_maps_yolov5s'
            save_path = os.path.join(save_dir, f'mscoco_{method}_all_conv.pickle')

            # if os.path.exists(save_path) and method != 'OTD': continue

            logging.info(f"Start {method}")
            print(f"Start {method}")

            results_all = {
                "DET vs FullGradCam":defaultdict(defaultdict),
                "EXP vs ODAM":defaultdict(defaultdict),
                "PV vs FullGradCam":defaultdict(defaultdict),
            }

            for layer in tqdm.tqdm(layer_name_mapping):
                for img in human_attention['DET'].keys():
                    if img in all_failed_imgs: continue

                    if img in xai_saliency_maps['FullGradCAM'][layer].keys():
                        # DET vs FullGradCam
                        results_all['DET vs FullGradCam'][layer][img] = func(xai_saliency_maps['FullGradCAM'][layer][img], human_attention['DET'][img])
                        # PV vs FullGradCam
                        results_all['PV vs FullGradCam'][layer][img] = func(xai_saliency_maps['FullGradCAM'][layer][img], human_attention['PV'][img])
                    if img in xai_saliency_maps['ODAM'][layer].keys():
                        # EXP vs ODAM
                        results_all['EXP vs ODAM'][layer][img] = func(xai_saliency_maps['ODAM'][layer][img], human_attention['EXP'][img])

            os.makedirs(save_dir,exist_ok=True)
            pickle.dump(results_all, open(save_path, 'wb'))

            logging.info(f"Finish {method}")