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

    layers = ['backbone.res2.0.conv1','backbone.res2.0.conv2','backbone.res2.0.conv3','backbone.res2.1.conv1','backbone.res2.1.conv2','backbone.res2.1.conv3','backbone.res2.2.conv1','backbone.res2.2.conv2','backbone.res2.2.conv3','backbone.res3.0.conv1','backbone.res3.0.conv2','backbone.res3.0.conv3','backbone.res3.1.conv1','backbone.res3.1.conv2','backbone.res3.1.conv3','backbone.res3.2.conv1','backbone.res3.2.conv2','backbone.res3.2.conv3','backbone.res3.3.conv1','backbone.res3.3.conv2','backbone.res3.3.conv3','backbone.res4.0.conv1','backbone.res4.0.conv2','backbone.res4.0.conv3','backbone.res4.1.conv1','backbone.res4.1.conv2','backbone.res4.1.conv3','backbone.res4.2.conv1','backbone.res4.2.conv2','backbone.res4.2.conv3','backbone.res4.3.conv1','backbone.res4.3.conv2','backbone.res4.3.conv3','backbone.res4.4.conv1','backbone.res4.4.conv2','backbone.res4.4.conv3','backbone.res4.5.conv1','backbone.res4.5.conv2','backbone.res4.5.conv3','roi_heads.pooler.level_poolers.0','roi_heads.res5.0.conv1','roi_heads.res5.0.conv2','roi_heads.res5.0.conv3','roi_heads.res5.1.conv1','roi_heads.res5.1.conv2','roi_heads.res5.1.conv3','roi_heads.res5.2.conv1','roi_heads.res5.2.conv2','roi_heads.res5.2.conv3']

    result = cv2.imread(f"/home/jinhanz/cs/data/mscoco/images/resized/DET2/{target_img_name}.png")
                
    layer = 'roi_heads.res5.1.conv2'

    save_dir = f'/home/jinhanz/cs/xai/results/visualizations/241118_same_vs_different_fasterrcnn_{layer}/{condition}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fullgradcam_saliency_mask = torch.as_tensor(torch.load(os.path.join('/opt/jinhanz/results/bilinear/mscoco/xai_saliency_maps_fasterrcnn/',"fullgradcamraw",f"fullgradcamraw_COCO_NMS_class_{layer}_aifaith_norm_model_final_721ade_1",f"{target_img_name}{ext}.pth"))['masks_ndarray'])

    res_img = result.copy()
    res_img, _ = get_res_img(fullgradcam_saliency_mask.unsqueeze(0), res_img)
    res_img = res_img * 255
    cv2.imwrite(f"{save_dir}/{target_img_name}_fasterrcnn.png",res_img)

    human_attention = torch.as_tensor(scipy.io.loadmat(os.path.join('/home/jinhanz/cs/data/mscoco/human_attention/240107_DET_excluded_resized/attention_maps',f'{target_img_name}_GSmo_21.mat'))['output_map_norm'])

    res_img = result.copy()
    res_img, _ = get_res_img(human_attention.unsqueeze(0), res_img)
    res_img = res_img * 255
    cv2.imwrite(f"{save_dir}/{target_img_name}_human.png",res_img)

# head last layer           
imgs = {
    'similar':[
        "bird_100489",
        "refrigerator_536947",
        "carrot_287667",
        "backpack_177065",
        "cat_558073",
        "cake_119677",
        "backpack_370478",
        "baseball glove_162415",
        "teddy bear_82180",
        "teddy bear_205542",
        "zebra_449406",
        "toilet_42276",
        "dog_357459",
    ],
    "different":[
        "bed_491757",
        "dining table_480122",
        "bench_350607",
        "frisbee_139872",
        "laptop_482970",
        "skateboard_229553",
        "bowl_205834",
        "dining table_385029",
        "refrigerator_498463",
        "snowboard_393469",
    ]
}

# backbone last layer
imgs = {
    'similar':[
        "bird_100489",
        "refrigerator_536947",
        "backpack_370478",
        "baseball glove_162415",
        "teddy bear_205542", 
    ],
    "different":[
        "sandwich_465430",
        "bed_491757",
        "bottle_460929",
        "dining table_480122",
        "suitcase_350019",
        "bench_350607",
        "frisbee_139872",
        "cat_558073",
        "book_167159",
        "cell phone_480212",
        "wine glass_146489",
        "toilet_85576",
        "skateboard_229553",
    ]
}

# 5.1.conv2
imgs = {
    'similar':[
        "apple_216277",
        "giraffe_289659",
        "zebra_491613",
        "motorcycle_499622",
    ],
    "different":[
        "bear_519611",
        "bicycle_426166",
        "microwave_207538",
        "tie_244496",
        "skateboard_71877",
    ]
}

for condition, all_imgs in imgs.items():
    for target_img_name in all_imgs:
        ext = '-res.png'
        combine_images(condition,target_img_name,ext)