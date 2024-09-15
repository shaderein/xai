import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os, re, shutil

import cv2

def combine_images(root_dir, target_img_name, ext, sigma):

    model = 'yolov5s'
    title_font_size = 15

    layers = ['model_1_act', 'model_2_cv1_act', 'model_2_cv2_act', 'model_2_m_0_cv1_act', 'model_2_m_0_cv2_act', 'model_2_cv3_act', 'model_3_act', 'model_4_cv1_act', 'model_4_cv2_act', 'model_4_m_0_cv1_act', 'model_4_m_0_cv2_act', 'model_4_m_1_cv1_act', 'model_4_m_1_cv2_act', 'model_4_cv3_act', 'model_5_act', 'model_6_cv1_act', 'model_6_cv2_act', 'model_6_m_0_cv1_act', 'model_6_m_0_cv2_act', 'model_6_m_1_cv1_act', 'model_6_m_1_cv2_act', 'model_6_m_2_cv1_act', 'model_6_m_2_cv2_act', 'model_6_cv3_act', 'model_7_act', 'model_8_cv1_act', 'model_8_cv2_act', 'model_8_m_0_cv1_act', 'model_8_m_0_cv2_act', 'model_8_cv3_act', 'model_9_cv1_act', 'model_9_cv2_act', 'model_10_act', 'model_13_cv1_act', 'model_13_cv2_act', 'model_13_m_0_cv1_act', 'model_13_m_0_cv2_act', 'model_13_cv3_act', 'model_14_act', 'model_17_cv1_act', 'model_17_cv2_act', 'model_17_m_0_cv1_act', 'model_17_m_0_cv2_act', 'model_17_cv3_act']

    rows = 11
    cols = 4
    fig, axes = plt.subplots(rows,cols,figsize=(20,20))

    layer_idx = 1

    for i in range(rows):
        for j in range(cols):
            layer = layers[layer_idx-1]
            for dir in os.listdir(root_dir):
                if ".mat" in dir: continue
                if layer in dir:
                    image_path = os.path.join(root_dir,dir,f'{target_img_name}-res.{ext}')
                    image = cv2.imread(image_path)
                    if 'odam' in image_path:
                        if 'vehicle' in image_path or 'human' in image_path:
                            crop_img = image[:,round(image.shape[1]*1/3):round(image.shape[1]*2/3),:]
                        else:
                            crop_img = image[:,round(image.shape[1]*2/4):round(image.shape[1]*3/4),:]
                    elif 'fullgradcamraw' in image_path:
                        crop_img = image[:,round(image.shape[1]/2):,:]
                    ind_maps_dir = f"results/visualizations/all_maps_{root_dir.split('/')[-1]}/"
                    if not os.path.exists(ind_maps_dir):
                        os.makedirs(ind_maps_dir)
                    cv2.imwrite(os.path.join(ind_maps_dir,f"{layer}_{target_img_name}.png"),crop_img)

                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    axes[i,j].imshow(crop_img)
                    if "pooler" in layer:
                        axes[i,j].set_title(f"[{layer_idx}] {layer}",fontsize=int(title_font_size*0.8))
                    else:
                        axes[i,j].set_title(f"[{layer_idx}] {layer}",fontsize=title_font_size)
                    axes[i,j].axis('off')
                    layer_idx += 1
                    break

        #Hide any unused axes in the current row
        for j in range(len(layers[i]), cols):
            fig.delaxes(axes[i, j])

    plt.tight_layout()
    plt.savefig(f"results/visualizations/{model}_{root_dir.split('/')[-1]}_{sigma}_{target_img_name}.png")

# root_dir = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/xai_saliency_maps_faster/fullgradcamraw"
# target_img_name = 'chair_81061'
# ext = 'png'
# combine_images(root_dir,target_img_name,ext)

sigma = 'bilinear'
root_dir = f"/mnt/h/jinhan/results/mscoco/xai_saliency_maps_yolov5s_{sigma}/fullgradcamraw"
target_img_name = 'chair_81061'
ext = 'png'
combine_images(root_dir,target_img_name,ext,sigma)

# root_dir = "/mnt/h/OneDrive - The University Of Hong Kong/bdd/xai_saliency_maps_same_layer_faster/fullgradcamraw_human"
# target_img_name = '760'
# ext = 'jpg'
# combine_images(root_dir,target_img_name,ext)

# root_dir = "/mnt/h/OneDrive - The University Of Hong Kong/bdd/xai_saliency_maps_same_layer_faster/fullgradcamraw_human"
# target_img_name = '34'
# ext = 'jpg'
# combine_images(root_dir,target_img_name,ext)

# root_dir = "/mnt/h/OneDrive - The University Of Hong Kong/bdd/xai_saliency_maps_same_layer_faster/fullgradcamraw_vehicle"
# target_img_name = '940'
# ext = 'jpg'
# combine_images(root_dir,target_img_name,ext)

# root_dir = "/mnt/h/OneDrive - The University Of Hong Kong/bdd/xai_saliency_maps_same_layer_faster/fullgradcamraw_vehicle"
# target_img_name = '180'
# ext = 'jpg'
# combine_images(root_dir,target_img_name,ext)