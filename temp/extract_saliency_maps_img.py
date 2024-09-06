import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os, re, shutil

import cv2

def combine_images(root_dir, target_img_name, ext):

    model = 'fasterrcnn'
    title_font_size = 15

    layers = [
        ['backbone.res2.0.conv1',
        'backbone.res2.0.conv2',
        'backbone.res2.0.conv3',
        'backbone.res2.1.conv1',
        'backbone.res2.1.conv2',
        'backbone.res2.1.conv3',],
        ['backbone.res2.2.conv1',
        'backbone.res2.2.conv2',
        'backbone.res2.2.conv3',
        'backbone.res3.0.conv1',
        'backbone.res3.0.conv2',
        'backbone.res3.0.conv3',],
        ['backbone.res3.1.conv1',
        'backbone.res3.1.conv2',
        'backbone.res3.1.conv3',
        'backbone.res3.2.conv1',
        'backbone.res3.2.conv2',
        'backbone.res3.2.conv3',],
        ['backbone.res3.3.conv1',
        'backbone.res3.3.conv2',
        'backbone.res3.3.conv3',
        'backbone.res4.0.conv1',
        'backbone.res4.0.conv2',
        'backbone.res4.0.conv3',],
        ['backbone.res4.1.conv1',
        'backbone.res4.1.conv2',
        'backbone.res4.1.conv3',
        'backbone.res4.2.conv1',
        'backbone.res4.2.conv2',
        'backbone.res4.2.conv3',],
        ['backbone.res4.3.conv1',
        'backbone.res4.3.conv2',
        'backbone.res4.3.conv3',
        'backbone.res4.4.conv1',
        'backbone.res4.4.conv2',
        'backbone.res4.4.conv3',],
        ['backbone.res4.5.conv1',
        'backbone.res4.5.conv2',
        'backbone.res4.5.conv3',],

        #ROI
        ['roi_heads.pooler.level_poolers.0',],
        ['roi_heads.res5.0.conv1',
        'roi_heads.res5.0.conv2',
        'roi_heads.res5.0.conv3',
        'roi_heads.res5.1.conv1',
        'roi_heads.res5.1.conv2',
        'roi_heads.res5.1.conv3',],
        ['roi_heads.res5.2.conv1',
        'roi_heads.res5.2.conv2',
        'roi_heads.res5.2.conv3',]
    ]

    rows = len(layers)
    cols = max([len(c) for c in layers])
    fig, axes = plt.subplots(rows,cols,figsize=(20,20))

    layer_idx = 1

    for i in range(len(layers)):
        for j in range(len(layers[i])):
            layer = layers[i][j]
            for dir in os.listdir(root_dir):
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
                    ind_maps_dir = f"results/all_maps_{root_dir.split('/')[-1]}/"
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

        # Hide any unused axes in the current row
        for j in range(len(layers[i]), cols):
            fig.delaxes(axes[i, j])

    plt.tight_layout()
    plt.savefig(f"results/visualizations/{model}_{root_dir.split('/')[-1]}_{target_img_name}.png",dpi=1500)

# root_dir = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/xai_saliency_maps_faster/fullgradcamraw"
# target_img_name = 'chair_81061'
# ext = 'png'
# combine_images(root_dir,target_img_name,ext)

root_dir = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/xai_saliency_maps_faster/fullgradcamraw"
target_img_name = 'car_227511'
ext = 'png'
combine_images(root_dir,target_img_name,ext)

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