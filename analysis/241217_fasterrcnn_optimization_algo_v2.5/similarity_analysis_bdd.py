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

logging.basicConfig(filename='/home/jinhanz/cs/xai/logs/250105_fasterrcnn_correlation_process_bdd.log', 
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

for object in ['vehicle','human']: #,

    """
    human attention
    """
    if object == 'vehicle':
        skip_imgs = ['1007', '1023', '1028', '1041', '1079', '1108', '1121', '1127', '1170', '1201', '1253', '1258', '1272', '134', '1344', '1356', '210', '297', '321', '355', '383', '390', '406', '425', '485', '505', '52', '542', '634', '648', '711', '777', '784', '796', '797', '838', '848', '857', '899', '902', '953', '967', '969', '988', '99', '993']
        human_attention_path = {
                "DET":'/home/jinhanz/cs/data/bdd/human_attention/240107 Veh DET/whole_image',
                "EXP":'/home/jinhanz/cs/data/bdd/human_attention/240918 Veh EXP/human_saliency_map',
            }
    elif object == 'human':
        skip_imgs = ['2334', '1313', '1302', '2186', '1770', '1154', '1663', '186', '425', '875', '845', '829', '388', '748', '900', '1346', '1803', '1359', '1022', '97', '2203', '1066', '231', '1097', '488', '415', '2128', '2008', '2121', '2092', '2271', '1506', '1389', '1954', '2226', '670', '2161', '1041', '250', '1141', '348', '1063', '452', '601', '19', '1746', '1917', '1420', '1817', '270', '1398', '2040', '11', '1475', '897', '1805', '997', '1788']
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

    layer_name_mapping = ['backbone.res2.0.conv1','backbone.res2.0.conv2','backbone.res2.0.conv3','backbone.res2.1.conv1','backbone.res2.1.conv2','backbone.res2.1.conv3','backbone.res2.2.conv1','backbone.res2.2.conv2','backbone.res2.2.conv3','backbone.res3.0.conv1','backbone.res3.0.conv2','backbone.res3.0.conv3','backbone.res3.1.conv1','backbone.res3.1.conv2','backbone.res3.1.conv3','backbone.res3.2.conv1','backbone.res3.2.conv2','backbone.res3.2.conv3','backbone.res3.3.conv1','backbone.res3.3.conv2','backbone.res3.3.conv3','backbone.res4.0.conv1','backbone.res4.0.conv2','backbone.res4.0.conv3','backbone.res4.1.conv1','backbone.res4.1.conv2','backbone.res4.1.conv3','backbone.res4.2.conv1','backbone.res4.2.conv2','backbone.res4.2.conv3','backbone.res4.3.conv1','backbone.res4.3.conv2','backbone.res4.3.conv3','backbone.res4.4.conv1','backbone.res4.4.conv2','backbone.res4.4.conv3','backbone.res4.5.conv1','backbone.res4.5.conv2','backbone.res4.5.conv3','roi_heads.pooler.level','roi_heads.res5.0.conv1','roi_heads.res5.0.conv2','roi_heads.res5.0.conv3','roi_heads.res5.1.conv1','roi_heads.res5.1.conv2','roi_heads.res5.1.conv3','roi_heads.res5.2.conv1','roi_heads.res5.2.conv2','roi_heads.res5.2.conv3']
    for rescale_method in ['optimize_faithfulness_finer_v2.5','bilinear']: # ,
        for is_act in ["xai_saliency","rpn_saliency"]: #,

            if rescale_method == 'bilinear' and is_act == "rpn_activation":
                continue # the same as bilinear activation maps for normal (instead of RPN) condition
            
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
                "FullGradCAM":f'/opt/jinhanz/results/fasterrcnn/{rescale_method}/bdd/{is_act}_maps_fasterrcnn/fullgradcamraw_{object}',
                "ODAM":f'/opt/jinhanz/results/fasterrcnn/{rescale_method}/bdd/{is_act}_maps_fasterrcnn/odam_{object}',
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
                        try:
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

            similarity_methods = {
                # "RMSE": compute_RMSE,
                # "JSD": compute_jsd,
                "OTD": compute_otd,
                # "Correlation": compute_correlation,
                # "CosineSim": compute_cosine_similarity,
            }

            for method, func in similarity_methods.items():

                save_dir = f'/home/jinhanz/cs/xai/results/bdd/250105_{rescale_method}_{is_act}_maps_fasterrcnn'
                save_path = f'{save_dir}/{object}_{method}_all_conv.pickle'

                if os.path.exists(save_path) and method != 'OTD': continue

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