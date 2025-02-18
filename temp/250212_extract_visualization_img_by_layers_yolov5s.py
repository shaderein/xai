import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os, re, shutil, torch, tqdm, math, scipy.io

import numpy as np

import cv2

import warnings, logging
warnings.filterwarnings('ignore')

logging.basicConfig(filename='/home/jinhanz/cs/xai/logs/250212_extract_saliency_by_layers.log', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

layers = ['backbone.res2.0.conv1','backbone.res2.0.conv2','backbone.res2.0.conv3','backbone.res2.1.conv1','backbone.res2.1.conv2','backbone.res2.1.conv3','backbone.res2.2.conv1','backbone.res2.2.conv2','backbone.res2.2.conv3','backbone.res3.0.conv1','backbone.res3.0.conv2','backbone.res3.0.conv3','backbone.res3.1.conv1','backbone.res3.1.conv2','backbone.res3.1.conv3','backbone.res3.2.conv1','backbone.res3.2.conv2','backbone.res3.2.conv3','backbone.res3.3.conv1','backbone.res3.3.conv2','backbone.res3.3.conv3','backbone.res4.0.conv1','backbone.res4.0.conv2','backbone.res4.0.conv3','backbone.res4.1.conv1','backbone.res4.1.conv2','backbone.res4.1.conv3','backbone.res4.2.conv1','backbone.res4.2.conv2','backbone.res4.2.conv3','backbone.res4.3.conv1','backbone.res4.3.conv2','backbone.res4.3.conv3','backbone.res4.4.conv1','backbone.res4.4.conv2','backbone.res4.4.conv3','backbone.res4.5.conv1','backbone.res4.5.conv2','backbone.res4.5.conv3','roi_heads.pooler.level_poolers.0','roi_heads.res5.0.conv1','roi_heads.res5.0.conv2','roi_heads.res5.0.conv3','roi_heads.res5.1.conv1','roi_heads.res5.1.conv2','roi_heads.res5.1.conv3','roi_heads.res5.2.conv1','roi_heads.res5.2.conv2','roi_heads.res5.2.conv3']

def get_res_img(mask, res_img):
    mask = mask.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = (heatmap/255).astype(np.float32)
    #n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    res_img = (res_img / 255).astype(np.float32)
    res_img = cv2.add(res_img, heatmap)
    res_img = (res_img / res_img.max())
    return res_img, heatmap

def combine_images(is_act,rescale_method, dataset, layer_idx, layer, root_saved_dir):
    if dataset == 'COCO':
        root_dir = f"/opt/jinhanz/results/{rescale_method}/mscoco/{is_act}_maps_fasterrcnn/"
        model = 'model_final_721ade_1'
    elif dataset == 'vehicle':
        root_dir = f"/opt/jinhanz/results/{rescale_method}/bdd/{is_act}_maps_fasterrcnn/"
        model = 'FasterRCNN_C4_BDD100K_1'
    elif dataset == 'human':
        root_dir = f"/opt/jinhanz/results/{rescale_method}/bdd/{is_act}_maps_fasterrcnn/"
        model = 'FasterRCNN_C4_BDD100K_1'

    saved_dir = os.path.join(root_saved_dir, dataset)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    saved_path = os.path.join(saved_dir,f"[{layer_idx+1:02}]_{rescale_method}_{is_act}_{layer}.png")
    if os.path.exists(saved_path): return

    if dataset == 'COCO':
        attention_dir = os.path.join(root_dir,"fullgradcamraw",f"fullgradcamraw_{dataset}_NMS_class_{layer}_aifaith_norm_{model}")
    else:
        attention_dir = os.path.join(root_dir,f"fullgradcamraw_{dataset}",f"fullgradcamraw_{dataset}_NMS_class_{layer}_aifaith_norm_{model}")

    all_imgs = [f for f in os.listdir(attention_dir) if '.pth' in f ]
    all_imgs.sort()

    rows = math.ceil(math.sqrt(len(all_imgs)))
    cols = math.ceil(len(all_imgs) / rows)

    fig, axes = plt.subplots(rows,cols,figsize=(20,int(20*cols/rows)))

    img_idx = 0

    for i in range(rows):
        for j in range(cols):
            if img_idx >= len(all_imgs): break

            image_name = all_imgs[img_idx].replace('.pth','').replace('-res','')
            
            if dataset == 'COCO':
                result = cv2.imread(f"/home/jinhanz/cs/data/mscoco/images/resized/DET2/{image_name}")
            elif dataset == 'vehicle':
                result = cv2.imread(f"/home/jinhanz/cs/data/bdd/orib_veh_id_task0922/{image_name}")
            elif dataset == 'human':
                result = cv2.imread(f"/home/jinhanz/cs/data/bdd/orib_hum_id_task1009/{image_name}")

            fullgradcam_saliency_mask = torch.as_tensor(torch.load(os.path.join(attention_dir,all_imgs[img_idx]))['masks_ndarray'])
            # odam_saliency_mask = torch.as_tensor(torch.load(os.path.join(root_dir,"odam",f"odam_COCO_NMS_class_{layer}_aifaith_norm_model_final_721ade_1",f"{target_img_name}{ext}.pth"))['masks_ndarray'])

            res_img = result.copy()
            res_img, _ = get_res_img(fullgradcam_saliency_mask.unsqueeze(0), res_img)
            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            axes[i,j].imshow(res_img, extent=[0,result.shape[1],result.shape[0],0])
            axes[i,j].set_title(image_name,fontsize=7)
            
            axes[i,j].axis('off')
            img_idx += 1

    for i in range(img_idx, rows * cols + 1):
        fig.delaxes(axes.flatten()[i - 1])

    plt.tight_layout()
    plt.savefig(saved_path,dpi=300)

    # # Human attention

    # tasks = ['DET','EXP']

    # if dataset == 'COCO':
    #     tasks += ['PV']

    # for task in tasks:

    #     saved_path = os.path.join(saved_dir,f"{rescale_method}_{is_act}_{task}.png")
    #     if os.path.exists(saved_path): return

    #     if dataset == 'COCO':
    #         if task == "DET":
    #             attention_dir = '/home/jinhanz/cs/data/mscoco/human_attention/240107_DET_excluded_resized/attention_maps'
    #         if task == "EXP":
    #             attention_dir = '/home/jinhanz/cs/data/mscoco/human_attention/231222_EXP_excluded_cleaned_resized/attention_maps'
    #         if task == "PV":
    #             attention_dir = '/home/jinhanz/cs/data/mscoco/human_attention/231221_PV_resized/attention_maps'
    #     elif dataset == 'vehicle':
    #         if task == "DET":
    #             attention_dir = '/home/jinhanz/cs/data/bdd/human_attention/240107 Veh DET/whole_image'
    #         if task == "EXP":
    #             attention_dir = '/home/jinhanz/cs/data/bdd/human_attention/240918 Veh EXP/human_saliency_map'
    #     elif dataset == 'human':
    #         if task == "DET":
    #             attention_dir = '/home/jinhanz/cs/data/bdd/human_attention/240107 Hum DET/whole_image'
    #         if task == "EXP":
    #             attention_dir = '/home/jinhanz/cs/data/bdd/human_attention/240918 Hum EXP/human_saliency_map'

    #     fig, axes = plt.subplots(rows,cols,figsize=(40,int(40*cols/rows)))

    #     img_idx = 0

    #     for i in range(rows):
    #         for j in range(cols):
    #             if img_idx >= len(all_imgs): break

    #             image_name = all_imgs[img_idx].replace('.pth','').replace('-res','')
                
    #             if dataset == 'COCO':
    #                 result = cv2.imread(f"/home/jinhanz/cs/data/mscoco/images/resized/DET2/{image_name}")
    #                 img_attention_path = os.path.join(attention_dir,image_name.replace('.png','')+'_GSmo_21.mat')
    #             elif dataset == 'vehicle':
    #                 result = cv2.imread(f"/home/jinhanz/cs/data/bdd/orib_veh_id_task0922/{image_name}")
    #                 img_attention_path = os.path.join(attention_dir,image_name.replace('.jpg','')+'_GSmo_30.mat')
    #             elif dataset == 'human':
    #                 result = cv2.imread(f"/home/jinhanz/cs/data/bdd/orib_hum_id_task1009/{image_name}")
    #                 img_attention_path = os.path.join(attention_dir,image_name.replace('.jpg','')+'_GSmo_30.mat')

    #             human_attention = torch.as_tensor(scipy.io.loadmat(img_attention_path)['output_map_norm'])

    #             res_img = result.copy()
    #             res_img, _ = get_res_img(human_attention.unsqueeze(0), res_img)
    #             res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    #             axes[i,j].imshow(res_img, extent=[0,result.shape[1],result.shape[0],0])
    #             axes[i,j].set_title(image_name)
                
    #             axes[i,j].axis('off')
    #             img_idx += 1

    #     for i in range(img_idx, rows * cols + 1):
    #         fig.delaxes(axes.flatten()[i - 1])

    #     plt.tight_layout()
    #     plt.savefig(saved_path,dpi=300)

root_saved_dir = f"results/visualizations/241210_finer_faster_by_layers_v2/"

for l,layer in zip(reversed(range(len(layers))), reversed(layers)):
    for dataset in ['human']: #'COCO','vehicle',
        for is_act in ['xai_saliency']:
            for rescale_method in ['optimize_faithfulness_finer']:
                # try:
                logging.info(f"{dataset} {layer} started")
                combine_images(is_act,rescale_method,dataset,l,layer, root_saved_dir)
                # except:
                #     logging.exception(f"Error processing {dataset} {layer}")