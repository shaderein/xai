import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os, re, shutil, torch

import numpy as np

import cv2

def get_res_img(mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = (heatmap/255).astype(np.float32)
    #n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    res_img = (res_img / 255).astype(np.float32)
    res_img = cv2.add(res_img, heatmap)
    res_img = (res_img / res_img.max())
    return res_img, heatmap

def combine_images(root_dir, target_img_name, ext):

    model = 'yolov5s'
    title_font_size = 15

    layers = ['model_0_act','model_1_act', 'model_2_cv1_act', 'model_2_cv2_act', 'model_2_m_0_cv1_act', 'model_2_m_0_cv2_act', 'model_2_cv3_act', 'model_3_act', 'model_4_cv1_act', 'model_4_cv2_act', 'model_4_m_0_cv1_act', 'model_4_m_0_cv2_act', 'model_4_m_1_cv1_act', 'model_4_m_1_cv2_act', 'model_4_cv3_act', 'model_5_act', 'model_6_cv1_act', 'model_6_cv2_act', 'model_6_m_0_cv1_act', 'model_6_m_0_cv2_act', 'model_6_m_1_cv1_act', 'model_6_m_1_cv2_act', 'model_6_m_2_cv1_act', 'model_6_m_2_cv2_act', 'model_6_cv3_act', 'model_7_act', 'model_8_cv1_act', 'model_8_cv2_act', 'model_8_m_0_cv1_act', 'model_8_m_0_cv2_act', 'model_8_cv3_act', 'model_9_cv1_act', 'model_9_cv2_act', 'model_10_act', 'model_13_cv1_act', 'model_13_cv2_act', 'model_13_m_0_cv1_act', 'model_13_m_0_cv2_act', 'model_13_cv3_act', 'model_14_act', 'model_17_cv1_act', 'model_17_cv2_act', 'model_17_m_0_cv1_act', 'model_17_m_0_cv2_act', 'model_17_cv3_act']

    rows = 4
    cols = 11
    fig, axes = plt.subplots(rows*2,cols,figsize=(40,20))

    layer_idx = 1

    result = cv2.imread(f"/home/jinhanz/cs/data/mscoco/images/resized/DET2/{target_img_name}.png")

    raw_results = torch.load(os.path.join(root_dir,f"{target_img_name}.{ext}"))

    for i in range(rows):
        for j in range(cols):
            layer = layers[layer_idx-1]

            row = i * 2

            theoretical_saliency_mask = raw_results['theoretical_saliency_masks_all'][layer]
            effective_saliency_mask = raw_results['effective_saliency_masks_all'][layer]
            saliency_maps_orig_mapped = raw_results['saliency_maps_orig_mapped_all'][layer]

            # TRF
            res_img = result.copy()
            res_img, _ = get_res_img(sum(theoretical_saliency_mask).float(), res_img)
            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            axes[row,j].imshow(res_img, extent=[0,result.shape[1],result.shape[0],0])

            non_zero_indices = torch.nonzero(sum(saliency_maps_orig_mapped))
            axes[row,j].scatter(non_zero_indices[:, 3].cpu(), non_zero_indices[:, 2].cpu(), s=1, c='green', marker='o')

            res_img = result.copy()
            res_img, _ = get_res_img(sum(effective_saliency_mask).float(), res_img)
            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            axes[row+1,j].imshow(res_img, extent=[0,result.shape[1],result.shape[0],0])

            non_zero_indices = torch.nonzero(sum(saliency_maps_orig_mapped))
            axes[row+1,j].scatter(non_zero_indices[:, 3].cpu(), non_zero_indices[:, 2].cpu(), s=1, c='green', marker='o')
            
            if "pooler" in layer:
                axes[row,j].set_title(f"[{layer_idx}] {layer}",fontsize=int(title_font_size*0.8))
            else:
                axes[row,j].set_title(f"[{layer_idx}] {layer}",fontsize=title_font_size)
                axes[row+1,j].set_title("ERF",fontsize=title_font_size)
            axes[row,j].axis('off')
            axes[row+1,j].axis('off')
            layer_idx += 1

        #Hide any unused axes in the current row
        for j in range(len(layers[i]), cols):
            fig.delaxes(axes[i, j])

    plt.tight_layout()
    plt.savefig(f"results/visualizations/{condition}_{model}_raw_trfvserf_mapped_{target_img_name}.png")

# root_dir = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/xai_saliency_maps_faster/fullgradcamraw"
# target_img_name = 'chair_81061'
# ext = 'png'
# combine_images(root_dir,target_img_name,ext)

# for sigma in ['bilinear','gaussian_sigma2','gaussian_sigma4']:
#     root_dir = f"/opt/jinhanz/results/bdd/xai_saliency_maps_yolov5s_{sigma}/fullgradcamraw_vehicle"
#     target_img_name = '117'
#     ext = 'jpg'
#     combine_images(root_dir,target_img_name,ext,sigma)

condition = "perturb_pixel_whole_optimal_sigma"

root_dir = f"/opt/jinhanz/results/{condition}/mscoco/xai_saliency_maps_yolov5s_raw_masks/"
target_img_name = 'chair_81061'
ext = 'pth'
combine_images(root_dir,target_img_name,ext)