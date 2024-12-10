import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os, re, shutil, torch, tqdm, scipy.io

import numpy as np

import cv2

def get_res_img(mask, res_img):
    mask = mask.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = (heatmap/255).astype(np.float32)
    #n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    res_img = (res_img / 255).astype(np.float32)
    res_img = cv2.add(res_img, heatmap)
    res_img = (res_img / res_img.max())
    return res_img, heatmap

def combine_images(condition,target_img_name, ext):

    layers = ['model_1_act', 'model_2_cv1_act', 'model_2_cv2_act', 'model_2_m_0_cv1_act', 'model_2_m_0_cv2_act', 'model_2_cv3_act', 'model_3_act', 'model_4_cv1_act', 'model_4_cv2_act', 'model_4_m_0_cv1_act', 'model_4_m_0_cv2_act', 'model_4_m_1_cv1_act', 'model_4_m_1_cv2_act', 'model_4_cv3_act', 'model_5_act', 'model_6_cv1_act', 'model_6_cv2_act', 'model_6_m_0_cv1_act', 'model_6_m_0_cv2_act', 'model_6_m_1_cv1_act', 'model_6_m_1_cv2_act', 'model_6_m_2_cv1_act', 'model_6_m_2_cv2_act', 'model_6_cv3_act', 'model_7_act', 'model_8_cv1_act', 'model_8_cv2_act', 'model_8_m_0_cv1_act', 'model_8_m_0_cv2_act', 'model_8_cv3_act', 'model_9_cv1_act', 'model_9_cv2_act', 'model_10_act', 'model_13_cv1_act', 'model_13_cv2_act', 'model_13_m_0_cv1_act', 'model_13_m_0_cv2_act', 'model_13_cv3_act', 'model_14_act', 'model_17_cv1_act', 'model_17_cv2_act', 'model_17_m_0_cv1_act', 'model_17_m_0_cv2_act', 'model_17_cv3_act']

    result = cv2.imread(f"/home/jinhanz/cs/data/mscoco/images/resized/DET2/{target_img_name}.png")
                
    layer = 'model_8_cv3_act'

    save_dir = f'/home/jinhanz/cs/xai/results/visualizations/241118_same_vs_different_yolov5s_{layer}/{condition}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fullgradcam_saliency_mask = torch.as_tensor(scipy.io.loadmat(os.path.join('/opt/jinhanz/results/perturb_pixel_whole/mscoco/xai_saliency_maps_yolov5s_bilinear',"fullgradcamraw",f"fullgradcamraw_COCO_NMS_class_{layer}_aifaith_norm_yolov5s_COCOPretrained_1",f"{target_img_name}{ext}.mat"))['masks_ndarray'])

    res_img = result.copy()
    res_img, _ = get_res_img(fullgradcam_saliency_mask.unsqueeze(0), res_img)
    res_img = res_img * 255
    cv2.imwrite(f"{save_dir}/{target_img_name}_yolov5s.png",res_img)

    human_attention = torch.as_tensor(scipy.io.loadmat(os.path.join('/home/jinhanz/cs/data/mscoco/human_attention/240107_DET_excluded_resized/attention_maps',f'{target_img_name}_GSmo_21.mat'))['output_map_norm'])

    res_img = result.copy()
    res_img, _ = get_res_img(human_attention.unsqueeze(0), res_img)
    res_img = res_img * 255
    cv2.imwrite(f"{save_dir}/{target_img_name}_human.png",res_img)

# backbone last layer
imgs = {
    'similar':[
        "bottle_385029",
        "cow_361268",
        "keyboard_378099",
        "remote_476810",
        "teddy bear_82180",
        "traffic light_133087",
        "truck_334006",
        "tv_104666",
        "tv_453722",
    ],
    "different":[
        "banana_279769",
        "boat_178744",
        "broccoli_389381",
        "broccoli_61658",
        "bus_226154",
        "couch_31735",
        "fire hydrant_293071",
        "horse_382088",
        "orange_50679",
        "pizza_294831",
        "refrigerator_498463",
        "sports ball_60102",
        "suitcase_350019",
        "tie_244496",
        "wine glass_146489",
        "skateboard_229553"
    ]
}

# # neck last layer
# imgs = {
#     'similar':[
#         "airplane_167540",
#         "bear_521231",
#         "refrigerator_536947",
#         "skateboard_71877",
#         "suitcase_23023",
#     ],
#     "different":[
#         "bird_404568",          
#         "frisbee_139872",       
#         "sandwich_417608",      
#         "wine glass_146489",    
#         "wine glass_25394",     
#     ]
# }

for condition, all_imgs in imgs.items():
    for target_img_name in all_imgs:
        ext = '-res.png'
        combine_images(condition,target_img_name,ext)