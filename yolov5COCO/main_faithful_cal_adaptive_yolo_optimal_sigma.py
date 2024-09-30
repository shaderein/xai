import warnings
warnings.filterwarnings("ignore")

import os,re
import time
import argparse
import numpy as np
from models.xai_method_optimal_sigma import YOLOV5XAI
# from models.eigencam import YOLOV5EigenCAM
# from models.eigengradcam import YOLOV5EigenGradCAM
# from models.weightedgradcam import YOLOV5WeightedGradCAM
# from models.gradcamplusplus import YOLOV5GradCAMpp
# from models.fullgradcam import YOLOV5FullGradCAM
# from models.fullgradcamsqsq import YOLOV5FullGradCAMsqsq
# from models.fullgradcamraw import YOLOV5FullGradCAMraw
# from models.fullgradcampp import YOLOV5FullGradCAMpp
from models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
import cv2
from deep_utils import Box, split_extension
import scipy.io
import torch
# from numba import cuda
from GPUtil import showUtilization as gpu_usage
import gc
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math
import utils.util_my_yolov5 as ut

import torch.utils.data
# from torch.utils.tensorboard import SummaryWriter
#
# import test  # import test.py to get mAP after each epoch
# from models.yolo import Model
# from utils import google_utils
from utils.datasets import *
from utils.utils import *

import logging
import imageio

import pandas as pd

# Hyperparameter Settings
target_layer_group_dict = {
    "model_0_act" : ([6,2,2],[]),
    
    "model_1_act": ([3,2,1],['model_0_act']),
    "model_2" : {
        "model_2_cv1_act":([1,1,0],['model_1_act']),
        "model_2_cv2_act":([1,1,0],['model_1_act']),    
        },
    "model_2_m_0_cv1_act":([1,1,0],['model_2_cv1_act']),
    "model_2_m_0_cv2_act":([3,1,1],['model_2_m_0_cv1_act']),
    "model_2_cv3_act":([1,1,0],['model_2_cv2_act','model_2_m_0_cv2_act']),

    "model_3_act": ([3,2,1],['model_2_cv3_act']),
    "model_4" : {
        "model_4_cv1_act":([1,1,0],['model_3_act']),
        "model_4_cv2_act":([1,1,0],['model_3_act']),    
        },
    "model_4_m_0_cv1_act":([1,1,0],['model_4_cv1_act']),
    "model_4_m_0_cv2_act":([3,1,1],['model_4_m_0_cv1_act']),
    "model_4_m_1_cv1_act":([1,1,0],['model_4_m_0_cv2_act']),
    "model_4_m_1_cv2_act":([3,1,1],['model_4_m_1_cv1_act']),
    "model_4_cv3_act":([1,1,0],['model_4_cv2_act','model_4_m_1_cv2_act']),

    "model_5_act": ([3,2,1],['model_4_cv3_act']),
    "model_6" : {
        "model_6_cv1_act":([1,1,0],['model_5_act']),
        "model_6_cv2_act":([1,1,0],['model_5_act']),    
        },
    "model_6_m_0_cv1_act":([1,1,0],['model_6_cv1_act']),
    "model_6_m_0_cv2_act":([3,1,1],['model_6_m_0_cv1_act']),
    "model_6_m_1_cv1_act":([1,1,0],['model_6_m_0_cv2_act']),
    "model_6_m_1_cv2_act":([3,1,1],['model_6_m_1_cv1_act']),
    "model_6_m_2_cv1_act":([1,1,0],['model_6_m_1_cv2_act']),
    "model_6_m_2_cv2_act":([3,1,1],['model_6_m_2_cv1_act']),
    "model_6_cv3_act":([1,1,0],['model_6_cv2_act','model_6_m_2_cv2_act']),

    "model_7_act": ([3,2,1],['model_6_cv3_act']),
    "model_8" : {
        "model_8_cv1_act":([1,1,0],['model_7_act']),
        "model_8_cv2_act":([1,1,0],['model_7_act']),    
        },
    "model_8_m_0_cv1_act":([1,1,0],["model_8_cv1_act"]),
    "model_8_m_0_cv2_act":([3,1,1],["model_8_m_0_cv1_act"]),
    "model_8_cv3_act":([1,1,0],['model_8_cv2_act',"model_8_m_0_cv2_act"]),

    # SPPF
    "model_9_cv1_act":([1,1,0],['model_8_cv3_act']),
    "model_9_m_act1":([5,1,2],[]), # repeated # skip pooling layer when calculating ERF
    "model_9_m_act2":([5,1,2],[]),
    "model_9_m_act3":([5,1,2],[]),
    "model_9_cv2_act":([1,1,0],['model_9_cv1_act']),

    "model_10_act":([1,1,0],['model_9_cv2_act']),
    "model_13" : {
        "model_13_cv1_act":([1,1,0],['model_6_cv3_act','model_10_act']),
        "model_13_cv2_act":([1,1,0],['model_6_cv3_act','model_10_act']),    
        },
    "model_13_m_0_cv1_act":([1,1,0],['model_13_cv1_act']),
    "model_13_m_0_cv2_act":([3,1,1],['model_13_m_0_cv1_act']),
    "model_13_cv3_act":([1,1,0],['model_13_cv2_act','model_13_m_0_cv2_act']),

    "model_14_act":([1,1,0],['model_13_cv3_act']),
    "model_17" : {
        "model_17_cv1_act":([1,1,0],['model_4_cv3_act','model_14_act']),
        "model_17_cv2_act":([1,1,0],['model_4_cv3_act','model_14_act']),    
        },
    "model_17_m_0_cv1_act":([1,1,0],['model_17_cv1_act']),
    "model_17_m_0_cv2_act":([3,1,1],['model_17_m_0_cv1_act']),
    "model_17_cv3_act":([1,1,0],['model_17_cv2_act','model_17_m_0_cv2_act']),

    "model_18_act":([3,2,1],['model_17_cv3_act']),
    "model_20" : {
        "model_20_cv1_act":([1,1,0],['model_14_act','model_18_act']),
        "model_20_cv2_act":([1,1,0],['model_14_act','model_18_act']),    
        },
    "model_20_m_0_cv1_act":([1,1,0],['model_20_cv1_act']),
    "model_20_m_0_cv2_act":([3,1,1],['model_20_m_0_cv1_act']),
    "model_20_cv3_act":([1,1,0],['model_20_cv2_act','model_20_m_0_cv2_act']),

    "model_21_act":([3,2,1],['model_20_cv3_act']),
    "model_23" : {
        "model_23_cv1_act":([1,1,0],['model_10_act','model_21_act']),
        "model_23_cv2_act":([1,1,0],['model_10_act','model_21_act']),    
        },
    "model_23_m_0_cv1_act":([1,1,0],['model_23_cv1_act']),
    "model_23_m_0_cv2_act":([3,1,1],['model_23_m_0_cv1_act']),
    "model_23_cv3_act":([1,1,0],['model_23_cv2_act','model_23_m_0_cv2_act']),    
}

excluded_layers = ["model_9_m_act1","model_9_m_act2","model_9_m_act3"]

skipped_levels = {
    0 : ['model_23','model_21','model_20','model_18'],
    1 : ['model_23','model_21',],
    2 : []
}

# default class=vehicle
input_main_dir = 'orib_veh_id_task0922'   #COCO_YOLO_IMAGE  Veh_id_img orib_veh_id_task_previous orib_veh_id_task0922
input_main_dir_label = 'orib_veh_id_task0922_label'   #COCO_YOLO_LABEL  Veh_id_label orib_veh_id_task_previous_label orib_veh_id_task0922_label
output_main_dir = 'multi_layer_analysis/odam_test_results'

# sel_method = 'DRISE'  # gradcam, gradcampp, fullgradcam, fullgradcamraw, saveRawGradAct, saveRawAllAct, DRISE
sel_nms = 'NMS'     # non maximum suppression
# sel_prob = 'class'  # obj, class, objclass
sel_norm = 'norm'
sel_faith = 'aifaith'     # nofaith, aifaith, humanfaith, aihumanfaith, trainedXAIfaith


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default='yolov5sbdd100k300epoch.pt', help='Path to the model')
parser.add_argument('--img-path', type=str, default=input_main_dir, help='input image path')
parser.add_argument('--output-dir', type=str, default='sample_EM_idtask_1_output_update_2/GradCAM_NMS_objclass_F0_singleScale_norm_v5s_1', help='output dir')
parser.add_argument('--output-main-dir', type=str, default=output_main_dir, help='output root dir')
parser.add_argument('--img-size', type=int, default=608, help="input image size")
# parser.add_argument('--target-layer', type=list, default=list(target_layer_group_list[0]),
#                     help='The layer hierarchical address to which gradcam will applied,'
#                          ' the names should be separated by underline')

parser.add_argument('--method', type=str, default="gradcam", help='gradcam or eigencam or eigengradcam or weightedgradcam or gradcampp or fullgradcam')
parser.add_argument('--device', type=str, default='0', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
parser.add_argument('--label-path', type=str, default=input_main_dir_label, help='input label path')

parser.add_argument('--object', type=str, default="vehicle", help='human or vehicle')
parser.add_argument('--prob', type=str, default="class", help='obj, class, objclass')
parser.add_argument('--coco-labels', type=str, default="COCO_classes.txt", help='path to coco classes list')

parser.add_argument('--img-start', type=int, default=0)
parser.add_argument('--img-end', type=int, default=160)

gc.collect()
torch.cuda.empty_cache()
gpu_usage()

logging.basicConfig(filename="./whole_image.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


def area(a, b, threshold=0.5): 
    """
    a = prediction box, b = GT BB
    in the form of [y1,x1, y2,x2]

    return the max percentage of two below:
        1. intersect / area of a
        2. intersect / area of b
    returns -1 if rectangles don't intersect
    """
    dx = min(a[3], b[3]) - max(a[1], b[1])
    dy = min(a[2], b[2]) - max(a[0], b[0])
    if (dx>=0) and (dy>=0):
        overlap = dx*dy
        # For vehicle: prediction box is usually smaller than GT BB
        # For human: prediction box is ually larger than GT BB
        # Thus take the max to prevent missing the correct prediction box
        percentage = max(\
            overlap / ((a[3]-a[1])*(a[2]-a[0])),\
            overlap / ((b[3]-b[1])*(b[2]-b[0])))
        return  percentage if percentage > threshold else -1
    else:
        return -1

def bbox_iou(bbox1, bbox2):
    """
    计算两个边界框的交并比 (IoU) 值

    :param bbox1: 列表，包含四个 int 元素，代表 yxyx 格式的四个坐标
    :param bbox2: 列表，包含四个 int 元素，代表 yxyx 格式的四个坐标
    :return: float, 两个边界框的 IoU 值
    """
    # 计算交集区域的坐标
    x0 = max(bbox1[1], bbox2[1])
    y0 = max(bbox1[0], bbox2[0])
    x1 = min(bbox1[3], bbox2[3])
    y1 = min(bbox1[2], bbox2[2])

    # 计算交集区域的面积
    intersection_area = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)

    # 计算两个边界框的面积
    bbox1_area = (bbox1[3] - bbox1[1] + 1) * (bbox1[2] - bbox1[0] + 1)
    bbox2_area = (bbox2[3] - bbox2[1] + 1) * (bbox2[2] - bbox2[0] + 1)

    # 计算并集区域的面积
    union_area = bbox1_area + bbox2_area - intersection_area

    # 计算交并比 (IoU)
    iou = intersection_area / union_area

    return iou

def main(img_path, label_path, model, saliency_method, img_num, 
         class_names_sel,class_names_gt,
         args,
         sigma_factors=[2,4],
         save_visualization=False):
    gc.collect()
    torch.cuda.empty_cache()

    img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')

    img = cv2.imread(img_path)
    torch_img = model.preprocessing(img[..., ::-1])

    # Find instance used in experiments
    if args.object=='vehicle':
        bb_selections = pd.read_excel('/home/jinhanz/cs/data/bdd/labels_mapping/Random_sample_vehicle_procedure_analysis.xlsx','veh_sample_img_condition')
        bb_selections = bb_selections[['image','vehicle_count_gt','ExpTargetIndex']]
        bb_selection = bb_selections.loc[bb_selections['image']==img_path.split('/')[-1]] # 1029.jpg
    elif args.object=='human':
        bb_selections = pd.read_excel('/home/jinhanz/cs/data/bdd/labels_mapping/Random_sample_human_procedure_analysis.xlsx','hum_sample_img_condition')
        bb_selections = bb_selections[['imgnumber','human_count_gt','ExpTargetIndex']]
        bb_selection = bb_selections.loc[bb_selections['imgnumber']==int(img_path.split('/')[-1].replace('.jpg',''))] # 1029.jpg
    elif args.object == 'COCO':
        # Find instance used in experiments
        bb_selections = pd.read_excel('/home/jinhanz/cs/data/mscoco/other/for_eyegaze_GT_infos_size_ratio.xlsx')
        bb_selection = bb_selections.loc[bb_selections['img']==img_path.split('/')[-1].replace('.jpg','')] # horse_382088.png

    tic = time.time()

    masks_all, masks_sum_all, activations_all, activations_sum_all, masks_orig_all, masks_mapped_all, rf_masks, rf_sizes, [boxes, _, class_names, obj_prob], class_prob_list, head_num_list, raw_data = saliency_method(torch_img,(img.shape[0],img.shape[1]),sigma_factors)
    print("\n[Raw] total time:", round(time.time() - tic, 4))

    effective_saliency_masks_all, theoretical_saliency_masks_all,effective_activation_masks_all, theoretical_activation_masks_all,= rf_masks
    effective_saliency_rf_all, effective_activation_rf_all, theoretical_rf_all = rf_sizes

    saliency_maps_orig_all, activation_maps_orig_all = masks_orig_all
    saliency_maps_orig_mapped_all, activation_maps_orig_mapped_all = masks_mapped_all

    raw_masks = {
        "effective_saliency_masks_all" : effective_saliency_masks_all,
        "theoretical_saliency_masks_all" : theoretical_saliency_masks_all,
        "effective_activation_masks_all" : effective_activation_masks_all,
        "theoretical_activation_masks_all" : theoretical_activation_masks_all,
        "effective_saliency_rf_all" : effective_saliency_rf_all,
        "effective_activation_rf_all" : effective_activation_rf_all,
        "theoretical_rf_all" : theoretical_rf_all,
        "saliency_maps_orig_all" : saliency_maps_orig_all,
        "activation_maps_orig_all" : activation_maps_orig_all,
        "saliency_maps_orig_mapped_all" : saliency_maps_orig_mapped_all,
        "activation_maps_orig_mapped_all" : activation_maps_orig_mapped_all,
    }
    os.makedirs(args.output_main_dir.replace('optimal_gaussian_sigmaFACTOR/fullgradcamraw','raw_masks'), exist_ok=True)
    torch.save(raw_masks,os.path.join(args.output_main_dir.replace('optimal_gaussian_sigmaFACTOR/fullgradcamraw','raw_masks'),f"{img_path.split('/')[-1].split('.')[0]}.pth"))

    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr
    images = [result]

    ### New Images
    result_raw = result
    images = [img]
    result = img

    ### Rescale Boxes
    shape_raw = [np.size(result_raw, 1), np.size(result_raw, 0)]  # w, h
    shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
    boxes_rescale_xyxy, boxes_rescale_xywh, boxes = ut.rescale_box_list(boxes, shape_raw, shape_new)

    ### Load labels
    boxes_GT, label_data_corr_xyxy, label_data_corr_xywh, label_data_corr_yxyx, label_data_class, label_data_class_names\
        = ut.load_gt_labels(img, label_path, class_names_gt, class_names_sel)

    # ODAM Find exp instance
    if args.object == 'COCO':
        target_bbox_GT = [bb_selection[['y1']].y1.item()*img.shape[0],
                        bb_selection[['x1']].x1.item()*img.shape[1],
                        bb_selection[['y2']].y2.item()*img.shape[0],
                        bb_selection[['x2']].x2.item()*img.shape[1]]
        boxes_GT_overlaps = np.zeros(len(boxes_GT))
        for i, box_GT in enumerate(boxes_GT):
            iou = bbox_iou(box_GT[0],target_bbox_GT)
            boxes_GT_overlaps[i] = iou if iou > 0.1 else 0
        target_idx_GT = np.argmax(boxes_GT_overlaps)    
    # For BDD: we only know the target index. Get GT target BB coordiates from input annotations
    else:
        # Find best matching target BB's index from pred (y1,x1,y2,x2)
        indices_GT_sorted = np.concatenate(boxes_GT,axis=0)[:, 1].argsort() # use x1 (top-left) to determine order of target (left to right)
        target_idx_GT = indices_GT_sorted[bb_selection['ExpTargetIndex'].values[0]-1]
        target_bbox_GT = boxes_GT[target_idx_GT][0]

    overlaps = np.zeros(len(boxes))
    target_indices_pred = [] # May miss the target
    for i in range(len(boxes)):
        # overlaps[i] = area(boxes[i][0],target_bbox_GT,threshold=0.1)
        overlaps[i] = bbox_iou(boxes[i][0],target_bbox_GT)
        if overlaps[i] < 0.1: overlaps[i] = 0
    if len(overlaps) > 0 and overlaps.max() > 0:
        target_indices_pred.append(np.argmax(overlaps))

    if len(target_indices_pred)==0: 
        return True
    
    for layer in masks_all.keys():

        sub_dir_name = args.method + '_' + args.object + '_' + sel_nms + '_' + args.prob + '_' + layer + '_' + sel_faith + '_' + sel_norm + '_' + args.model_path.split('/')[-1][:-3] + '_' + '1'
        args.output_dir = os.path.join(args.output_main_dir, sub_dir_name)

        effective_saliency_mask, theoretical_saliency_mask = effective_saliency_masks_all[layer], theoretical_saliency_masks_all[layer]
        saliency_maps_orig_mapped = saliency_maps_orig_mapped_all[layer]

        for sigma_factor in masks_all[layer]:

            masks = masks_all[layer][sigma_factor]
            masks_sum = masks_sum_all[layer][sigma_factor]

            masks_orig = masks
            masks = [masks_sum]

            # FullGradCAM

            if sigma_factor == -1:
                output_path = f'{args.output_dir.replace("gaussian_sigmaFACTOR","bilinear")}/{img_name}'
                os.makedirs(args.output_dir.replace("gaussian_sigmaFACTOR", "bilinear"), exist_ok=True)
            else:
                output_path = f'{args.output_dir.replace("FACTOR",str(sigma_factor))}/{img_name}'
                os.makedirs(args.output_dir.replace("FACTOR",str(sigma_factor)), exist_ok=True)

            ### Calculate AI Performance
            if len(boxes):
                Vacc = ut.calculate_acc(boxes_rescale_xywh, label_data_corr_xywh)/len(boxes_GT)
            else:
                Vacc = 0

            ### Display
            if save_visualization:
                for i, mask in enumerate(masks):
                    res_img = result.copy()
                    res_img, heat_map = ut.get_res_img(mask, res_img)
                for i, (bbox, cls_name, obj_logit, class_prob, head_num) in enumerate(zip(boxes, class_names, obj_prob, class_prob_list, head_num_list)):
                    if cls_name[0] in class_names_sel:
                        #bbox, cls_name = boxes[0][i], class_names[0][i]
                        # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
                        res_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob*100)[:2] + ", " + str(head_num)[:1], res_img) / 255

                ## Display Ground Truth
                gt_img = result.copy()
                gt_img = gt_img / gt_img.max()
                for i, (bbox, cls_idx) in enumerate(zip(boxes_GT, label_data_class)):
                    cls_idx = np.int8(cls_idx)
                    if class_names_gt[cls_idx] in class_names_sel:
                        #bbox, cls_name = boxes[0][i], class_names[0][i]
                        # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
                        gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img) / 255

                # images.append(gt_img * 255)
                images = [gt_img * 255]
                images.append(res_img * 255)
                final_image = ut.concat_images(images)
                os.makedirs(args.output_dir, exist_ok=True)
                cv2.imwrite(output_path, final_image)

                # TRF
                plt.figure()
                res_img = result.copy()
                res_img, _ = ut.get_res_img(sum(theoretical_saliency_mask).float(), res_img)
                res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                plt.imshow(res_img, extent=[0,img.shape[1],img.shape[0],0])

                non_zero_indices = torch.nonzero(sum(saliency_maps_orig_mapped))
                plt.scatter(non_zero_indices[:, 3].cpu(), non_zero_indices[:, 2].cpu(), s=1, c='green', marker='o')
                plt.axis('off')
                plt.savefig(output_path.replace("-res","-res_trf"), bbox_inches='tight', pad_inches=0, transparent=True)
                
                # ERF
                plt.figure()
                res_img = result.copy()
                res_img, _ = ut.get_res_img(sum(effective_saliency_mask).float(), res_img)
                res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                plt.imshow(res_img, extent=[0,img.shape[1],img.shape[0],0])

                non_zero_indices = torch.nonzero(sum(saliency_maps_orig_mapped))
                plt.scatter(non_zero_indices[:, 3].cpu(), non_zero_indices[:, 2].cpu(), s=1, c='green', marker='o')
                plt.axis('off')
                plt.savefig(output_path.replace("-res","-res_erf"), bbox_inches='tight', pad_inches=0, transparent=True)

            gc.collect()
            torch.cuda.empty_cache()

            masks_ndarray = masks[0].squeeze().detach().cpu().numpy()

            start = time.time()
            # AI Saliency Map Computation
            preds_deletion, preds_insertation, _, imgs_deletion, imgs_insertion = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh, class_names_sel)

            # # compress gif
            # downscaled_ratio = 0.4
            # saved_size = [int(res_img.shape[1] * downscaled_ratio),
            #               int(res_img.shape[0] * downscaled_ratio)
            #             ]
            # saliency_preview = [cv2.resize((res_img * 255).astype('uint8')[...,::-1], saved_size) for i in range(6)]
            # imgs_deletion_new = saliency_preview + [cv2.resize(img_orig, saved_size) for img_orig in imgs_deletion]
            # imgs_insertion_new = saliency_preview + [cv2.resize(img_orig, saved_size) for img_orig in imgs_insertion]
            # if sigma_factor == -1:
            #     imageio.mimsave(f'{os.path.join(args.output_dir.replace("gaussian_sigmaFACTOR","bilinear"),img_name+".deletion")}.gif', imgs_deletion_new, fps=2)
            #     imageio.mimsave(f'{os.path.join(args.output_dir.replace("gaussian_sigmaFACTOR","bilinear"),img_name+".insertion")}.gif', imgs_insertion_new, fps=2)
            # else:
            #     imageio.mimsave(f'{os.path.join(args.output_dir.replace("FACTOR",str(sigma_factor)),img_name+".deletion")}.gif', imgs_deletion_new, fps=2)
            #     imageio.mimsave(f'{os.path.join(args.output_dir.replace("FACTOR",str(sigma_factor)),img_name+".insertion")}.gif', imgs_insertion_new, fps=2)
            # #

            # Saving
            scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                        # 'masks_ndarray_all': [mask.squeeze().detach().cpu().numpy() for mask in masks_orig],
                                                        'head_num_list':head_num_list,
                                                        'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                        'boxes_pred_xywh': boxes_rescale_xywh,
                                                        'boxes_gt_xywh': label_data_corr_xywh,
                                                        'boxes_gt_xyxy': label_data_corr_xyxy,
                                                        'HitRate': Vacc,
                                                        'preds_deletion': np.array(preds_deletion,dtype='object'),
                                                        'preds_insertation': np.array(preds_insertation,dtype='object'),
                                                        'boxes_pred_conf': obj_prob,
                                                        'boxes_pred_class_names': class_names,
                                                        'class_names_sel': class_names_sel,
                                                        'boxes_gt_classes_names': label_data_class_names,
                                                        'grad_act': raw_data,
                                                        'target_indices_pred': target_indices_pred,
                                                        })
            end = time.time()
            print(f'[{sigma_factor}]: ({round(end-start,4)}s) save mat to: {output_path}')

            # ODAM

            masks = masks_orig

            ### Calculate AI Performance
            if len(boxes): # NOTE: For ODAM only consider the GT bbox and prediced bbox for one specific target
                Vacc = ut.calculate_acc(boxes_rescale_xywh[target_indices_pred], label_data_corr_xywh[[target_idx_GT]]) # NOTE: only 1 "GT" target
            else:
                Vacc = 0

            odam_output_dir = args.output_dir.replace('fullgradcamraw','odam')
            if sigma_factor == -1:
                output_path = f'{odam_output_dir.replace("gaussian_sigmaFACTOR","bilinear")}/{img_name}'
                os.makedirs(odam_output_dir.replace("gaussian_sigmaFACTOR", "bilinear"), exist_ok=True)
            else:
                output_path = f'{odam_output_dir.replace("FACTOR",str(sigma_factor))}/{img_name}'
                os.makedirs(odam_output_dir.replace("FACTOR",str(sigma_factor)), exist_ok=True)

            ### Display

            target_prob = []

            if not save_visualization:                
                for i, (bbox, cls_name, obj_logit, class_prob, head_num) in enumerate(zip(boxes, class_names, obj_prob, class_prob_list, head_num_list)):
                    if cls_name[0] in class_names_sel:
                        if i in target_indices_pred:
                            target_prob.append([class_prob])

            if save_visualization:

                # Generate whole-image saliency maps as reference        
                all_mask_img = result.copy()
                all_mask_img, heat_map = ut.get_res_img(masks_sum, all_mask_img)

                res_img = result.copy()
                for i, mask in enumerate(masks):
                    if i in target_indices_pred:
                        res_img, heat_map = ut.get_res_img(mask, res_img)

                target_prob = []
                for i, (bbox, cls_name, obj_logit, class_prob, head_num) in enumerate(zip(boxes, class_names, obj_prob, class_prob_list, head_num_list)):
                    if cls_name[0] in class_names_sel:
                        if i in target_indices_pred:
                            res_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob*100)[:2] + ", " + str(head_num)[:1], res_img) / 255
                            all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob*100)[:2] + ", " + str(head_num)[:1], all_mask_img) / 255
                            
                            # FIXME: check EXP BB selection
                            res_img = ut.put_text_box(target_bbox_GT, "GT BB for EXP", res_img, color=(255,0,0)) / 255

                            target_prob.append([class_prob])
                        else:
                            all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob*100)[:2] + ", " + str(head_num)[:1], all_mask_img, color=(0,0,255)) / 255

                ## Display Ground Truth
                gt_img = result.copy()
                gt_img = gt_img / gt_img.max()
                for i, (bbox, cls_idx) in enumerate(zip(boxes_GT, label_data_class)):
                    cls_idx = np.int8(cls_idx)
                    if class_names_gt[cls_idx] in class_names_sel:
                        if i==target_idx_GT:
                            gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img,color=(0,255,0)) / 255
                        else:
                            gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img,color=(0,0,255)) / 255
                
                # images.append(gt_img * 255)
                if args.object == 'COCO':
                    images = [cv2.imread(f"/home/jinhanz/cs/data/mscoco/images/resized/EXP/{img_path.split('/')[-1]}")]
                else:
                    images = [cv2.imread(f"/home/jinhanz/cs/data/bdd/{args.object}_exp/{img_path.split('/')[-1]}")]
                images.append(gt_img * 255)

                # no matching prediction and therefore empty saliency maps
                if len(target_indices_pred) == 0:
                    images.append(res_img) # original image
                else:
                    images.append(res_img * 255)

                images.append(all_mask_img * 255)
                final_image = ut.concat_images(images)

                img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
                os.makedirs(odam_output_dir, exist_ok=True)
                cv2.imwrite(output_path, final_image)

                res_img = result.copy()
                res_img, heat_map = ut.get_res_img(theoretical_saliency_mask[target_indices_pred[0]].float(), res_img) #TODO: inspect specific object
                cv2.imwrite(output_path.replace("-res","-res_trf"), res_img * 255)
                
                # ERF
                res_img = result.copy()
                res_img, heat_map = ut.get_res_img(effective_saliency_mask[target_indices_pred[0]].float(), res_img)
                cv2.imwrite(output_path.replace("-res","-res_erf"), res_img * 255)

            gc.collect()
            torch.cuda.empty_cache()

            if len(target_indices_pred) == 0:
                masks_ndarray = np.zeros(masks[0].squeeze().detach().cpu().numpy().shape)
            else: 
                masks_ndarray = masks[target_indices_pred[0]].squeeze().detach().cpu().numpy()

            start = time.time()
            # AI Saliency Map Computation
            preds_deletion, preds_insertation, _, imgs_deletion, imgs_insertion = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh[[target_idx_GT]], class_names_sel)


            # # compress gif
            # downscaled_ratio = 0.4
            # saved_size = [int(res_img.shape[1] * downscaled_ratio),
            #               int(res_img.shape[0] * downscaled_ratio)
            #             ]
            # saliency_preview = [cv2.resize((res_img * 255).astype('uint8')[...,::-1], saved_size) for i in range(6)]
            # imgs_deletion_new = saliency_preview + [cv2.resize(img_orig, saved_size) for img_orig in imgs_deletion]
            # imgs_insertion_new = saliency_preview + [cv2.resize(img_orig, saved_size) for img_orig in imgs_insertion]
            # if sigma_factor == -1:
            #     imageio.mimsave(f'{os.path.join(args.output_dir.replace("gaussian_sigmaFACTOR","bilinear").replace("fullgradcamraw","odam"),img_name+".deletion")}.gif', imgs_deletion_new)
            #     imageio.mimsave(f'{os.path.join(args.output_dir.replace("gaussian_sigmaFACTOR","bilinear").replace("fullgradcamraw","odam"),img_name+".insertion")}.gif', imgs_insertion_new)
            # else:
            #     imageio.mimsave(f'{os.path.join(args.output_dir.replace("FACTOR",str(sigma_factor)).replace("fullgradcamraw","odam"),img_name+".deletion")}.gif', imgs_deletion_new)
            #     imageio.mimsave(f'{os.path.join(args.output_dir.replace("FACTOR",str(sigma_factor)).replace("fullgradcamraw","odam"),img_name+".insertion")}.gif', imgs_insertion_new, fps=2)
            # #

            # Saving
            scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                        'head_num_list':head_num_list[target_indices_pred],
                                                        'boxes_pred_xyxy': boxes_rescale_xyxy[target_indices_pred],
                                                        'boxes_pred_xywh': boxes_rescale_xywh[target_indices_pred],
                                                        'boxes_gt_xywh': label_data_corr_xywh[[target_idx_GT]],
                                                        'boxes_gt_xyxy': label_data_corr_xyxy[[target_idx_GT]],
                                                        'HitRate': Vacc,
                                                        'preds_deletion': np.array(preds_deletion,dtype='object'),
                                                        'preds_insertation': np.array(preds_insertation,dtype='object'),
                                                        'boxes_pred_conf': target_prob,
                                                        'boxes_pred_class_names': class_names,
                                                        'class_names_sel': class_names_sel,
                                                        'boxes_gt_classes_names': label_data_class_names,
                                                        'grad_act': raw_data,
                                                        'target_indices_pred': target_indices_pred,
                                                        })
            end = time.time()
            print(f'[{sigma_factor}]: ({round(end-start,4)}s) save mat to: {output_path}')

            # Activation maps

            masks = activations_all[layer][sigma_factor]
            masks_sum = activations_sum_all[layer][sigma_factor]

            masks_orig = masks
            masks = [masks_sum]

            # # Excluded mask if it's from a layer skipped by its proposing head
            # if len(head_num_list) > 0:
            #     masks_sum_included = []
            #     included_count = 0
            #     for m, head in zip(masks, head_num_list):
            #         if any([l for l in skipped_levels[head[0]] if l in target_layer_group_name]):
            #             masks_sum_included.append(m)
            #             included_count += 1
            #     if included_count > 0:
            #         masks_sum = torch.cat(masks_sum_included).mean(axis=0).unsqueeze(dim=0)
            #     else: # layer excluded, masks_sum should automatically be 0?
            #         return

            # FullGradCAM

            ### Calculate AI Performance
            if len(boxes):
                Vacc = ut.calculate_acc(boxes_rescale_xywh, label_data_corr_xywh)/len(boxes_GT)
            else:
                Vacc = 0

            activation_output_dir = args.output_dir.replace('gaussian','act_gaussian')
            if sigma_factor == -1:
                output_path = f'{activation_output_dir.replace("gaussian_sigmaFACTOR","bilinear")}/{img_name}'
                os.makedirs(activation_output_dir.replace("gaussian_sigmaFACTOR", "bilinear"), exist_ok=True)
            else:
                output_path = f'{activation_output_dir.replace("FACTOR",str(sigma_factor))}/{img_name}'
                os.makedirs(activation_output_dir.replace("FACTOR",str(sigma_factor)), exist_ok=True)

            ### Display
            if save_visualization:
                for i, mask in enumerate(masks):
                    res_img = result.copy()
                    res_img, heat_map = ut.get_res_img(mask, res_img)
                for i, (bbox, cls_name, obj_logit, class_prob, head_num) in enumerate(zip(boxes, class_names, obj_prob, class_prob_list, head_num_list)):
                    if cls_name[0] in class_names_sel:
                        #bbox, cls_name = boxes[0][i], class_names[0][i]
                        # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
                        res_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob*100)[:2] + ", " + str(head_num)[:1], res_img) / 255

                ## Display Ground Truth
                gt_img = result.copy()
                gt_img = gt_img / gt_img.max()
                for i, (bbox, cls_idx) in enumerate(zip(boxes_GT, label_data_class)):
                    cls_idx = np.int8(cls_idx)
                    if class_names_gt[cls_idx] in class_names_sel:
                        #bbox, cls_name = boxes[0][i], class_names[0][i]
                        # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
                        gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img) / 255

                # images.append(gt_img * 255)
                images = [gt_img * 255]
                images.append(res_img * 255)
                final_image = ut.concat_images(images)
                img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
                os.makedirs(activation_output_dir, exist_ok=True)
                cv2.imwrite(output_path, final_image)

            gc.collect()
            torch.cuda.empty_cache()

            masks_ndarray = masks[0].squeeze().detach().cpu().numpy()

            start = time.time()
            # AI Saliency Map Computation
            preds_deletion, preds_insertation, _, _, _ = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh, class_names_sel)
            # Saving
            scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                        # 'masks_ndarray_all': [mask.squeeze().detach().cpu().numpy() for mask in masks_orig],
                                                        'head_num_list':head_num_list,
                                                        'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                        'boxes_pred_xywh': boxes_rescale_xywh,
                                                        'boxes_gt_xywh': label_data_corr_xywh,
                                                        'boxes_gt_xyxy': label_data_corr_xyxy,
                                                        'HitRate': Vacc,
                                                        'preds_deletion': np.array(preds_deletion,dtype='object'),
                                                        'preds_insertation': np.array(preds_insertation,dtype='object'),
                                                        'boxes_pred_conf': obj_prob,
                                                        'boxes_pred_class_names': class_names,
                                                        'class_names_sel': class_names_sel,
                                                        'boxes_gt_classes_names': label_data_class_names,
                                                        'grad_act': raw_data,
                                                        'target_indices_pred': target_indices_pred,
                                                        })
            end = time.time()
            print(f'[{sigma_factor}]: ({round(end-start,4)}s) save mat to: {output_path}')

            # ODAM

            masks = masks_orig

            ### Calculate AI Performance
            if len(boxes): # NOTE: For ODAM only consider the GT bbox and prediced bbox for one specific target
                Vacc = ut.calculate_acc(boxes_rescale_xywh[target_indices_pred], label_data_corr_xywh[[target_idx_GT]]) # NOTE: only 1 "GT" target
            else:
                Vacc = 0


            odam_output_dir = args.output_dir.replace('fullgradcamraw','odam').replace('gaussian','act_gaussian')
            if sigma_factor == -1:
                output_path = f'{odam_output_dir.replace("gaussian_sigmaFACTOR","bilinear")}/{img_name}'
                os.makedirs(odam_output_dir.replace("gaussian_sigmaFACTOR", "bilinear"), exist_ok=True)
            else:
                output_path = f'{odam_output_dir.replace("FACTOR",str(sigma_factor))}/{img_name}'
                os.makedirs(odam_output_dir.replace("FACTOR",str(sigma_factor)), exist_ok=True)

            ### Display

            if save_visualization:

                # Generate whole-image saliency maps as reference        
                all_mask_img = result.copy()
                all_mask_img, heat_map = ut.get_res_img(masks_sum, all_mask_img)

                res_img = result.copy()
                for i, mask in enumerate(masks):
                    if i in target_indices_pred:
                        res_img, heat_map = ut.get_res_img(mask, res_img)

                target_prob = []
                for i, (bbox, cls_name, obj_logit, class_prob, head_num) in enumerate(zip(boxes, class_names, obj_prob, class_prob_list, head_num_list)):
                    if cls_name[0] in class_names_sel:
                        if i in target_indices_pred:
                            res_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob*100)[:2] + ", " + str(head_num)[:1], res_img) / 255
                            all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob*100)[:2] + ", " + str(head_num)[:1], all_mask_img) / 255
                            
                            # FIXME: check EXP BB selection
                            res_img = ut.put_text_box(target_bbox_GT, "GT BB for EXP", res_img, color=(255,0,0)) / 255

                            target_prob.append([class_prob])
                        else:
                            all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob*100)[:2] + ", " + str(head_num)[:1], all_mask_img, color=(0,0,255)) / 255

                ## Display Ground Truth
                gt_img = result.copy()
                gt_img = gt_img / gt_img.max()
                for i, (bbox, cls_idx) in enumerate(zip(boxes_GT, label_data_class)):
                    cls_idx = np.int8(cls_idx)
                    if class_names_gt[cls_idx] in class_names_sel:
                        if i==target_idx_GT:
                            gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img,color=(0,255,0)) / 255
                        else:
                            gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img,color=(0,0,255)) / 255
                
                # images.append(gt_img * 255)
                if args.object == 'COCO':
                    images = [cv2.imread(f"/home/jinhanz/cs/data/mscoco/images/resized/EXP/{img_path.split('/')[-1]}")]
                else:
                    images = [cv2.imread(f"/home/jinhanz/cs/data/bdd/{args.object}_exp/{img_path.split('/')[-1]}")]
                images.append(gt_img * 255)

                # no matching prediction and therefore empty saliency maps
                if len(target_indices_pred) == 0:
                    images.append(res_img) # original image
                else:
                    images.append(res_img * 255)

                images.append(all_mask_img * 255)
                final_image = ut.concat_images(images)

                img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
                os.makedirs(odam_output_dir, exist_ok=True)
                cv2.imwrite(output_path, final_image)

            gc.collect()
            torch.cuda.empty_cache()

            if len(target_indices_pred) == 0:
                masks_ndarray = np.zeros(masks[0].squeeze().detach().cpu().numpy().shape)
            else: 
                masks_ndarray = masks[target_indices_pred[0]].squeeze().detach().cpu().numpy()

            start = time.time()
            # AI Saliency Map Computation
            preds_deletion, preds_insertation, _, _, _ = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh[[target_idx_GT]], class_names_sel)
            # Saving
            scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                        'head_num_list':head_num_list[target_indices_pred],
                                                        'boxes_pred_xyxy': boxes_rescale_xyxy[target_indices_pred],
                                                        'boxes_pred_xywh': boxes_rescale_xywh[target_indices_pred],
                                                        'boxes_gt_xywh': label_data_corr_xywh[[target_idx_GT]],
                                                        'boxes_gt_xyxy': label_data_corr_xyxy[[target_idx_GT]],
                                                        'HitRate': Vacc,
                                                        'preds_deletion': np.array(preds_deletion,dtype='object'),
                                                        'preds_insertation': np.array(preds_insertation,dtype='object'),
                                                        'boxes_pred_conf': target_prob,
                                                        'boxes_pred_class_names': class_names,
                                                        'class_names_sel': class_names_sel,
                                                        'boxes_gt_classes_names': label_data_class_names,
                                                        'grad_act': raw_data,
                                                        'target_indices_pred': target_indices_pred,
                                                        })
            end = time.time()
            print(f'[{sigma_factor}]: ({round(end-start,4)}s) save mat to: {output_path}')

    return False


if __name__ == '__main__':

    args = parser.parse_args()

    if args.object == "COCO":
        args.model_path = "/home/jinhanz/cs/xai/models/yolov5s_COCOPretrained.pt"
        args.method = "fullgradcamraw"
        args.prob = "class"
        args.output_main_dir = "/opt/jinhanz/results/perturb_pixel_whole_optimal_sigma/mscoco/xai_saliency_maps_yolov5s_optimal_gaussian_sigmaFACTOR/fullgradcamraw"
        args.coco_labels = "/home/jinhanz/cs/data/mscoco/annotations/COCO_classes2.txt"
        args.img_path = "/home/jinhanz/cs/data/mscoco/images/resized/DET2"
        args.label_path = "/home/jinhanz/cs/data/mscoco/annotations/annotations_DET2"

    elif args.object == "vehicle":
        args.model_path = "/home/jinhanz/cs/xai/models/yolov5s_bdd_300epoch_240628_best.pt"
        args.method = "fullgradcamraw"
        args.prob = "class"
        args.output_main_dir = "/opt/jinhanz/results/perturb_pixel_whole_optimal_sigma/bdd/xai_saliency_maps_yolov5s_optimal_gaussian_sigmaFACTOR/fullgradcamraw_vehicle"
        args.img_path = "/home/jinhanz/cs/data/bdd/orib_veh_id_task0922"
        args.label_path = "/home/jinhanz/cs/data/bdd/orib_veh_id_task0922_label"

    elif args.object == "human":
        args.model_path = "/home/jinhanz/cs/xai/models/yolov5s_bdd_300epoch_240628_best.pt"
        args.method = "fullgradcamraw"
        args.prob = "class"
        args.output_main_dir = "/opt/jinhanz/results/perturb_pixel_whole_optimal_sigma/bdd/xai_saliency_maps_yolov5s_optimal_gaussian_sigmaFACTOR/fullgradcamraw_human"
        args.img_path = "/home/jinhanz/cs/data/bdd/orib_hum_id_task1009"
        args.label_path = "/home/jinhanz/cs/data/bdd/orib_hum_id_task1009_label"

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    device = f"cuda"

    # input_size = (args.img_size, args.img_size)
    save_visualization = True
    sigma_factors = [2,4]

    print(f'[INFO] {args}')
    print('[INFO] Loading the model')

    failed_imgs = []

    if args.object == 'COCO':
        input_size = (640, 640)
        sampled_images = ['elephant_97230.png','chair_81061.png','giraffe_287545.png']
        skip_images = ['book_472678.png',"baseball glove_515982.png","toothbrush_160666.png","potted plant_473219.png","bench_350607.png","truck_295420.png","toaster_232348.png","kite_405279.png","toothbrush_218439.png","snowboard_425906.png","car_227511.png","traffic light_453841.png","hair drier_239041.png","hair drier_178028.png","toaster_453302.png","mouse_513688.png","spoon_88040.png","scissors_340930.png","handbag_383842.png"]
    elif args.object == 'vehicle':
        input_size = (608, 608)
        sampled_images = ['362.jpg','930.jpg','117.jpg']
        skip_images = ["178.jpg", "54.jpg", "452.jpg", "478.jpg", "629.jpg", "758.jpg", "856.jpg",'1007.jpg', '1028.jpg', '1041.jpg', '1065.jpg', '1100.jpg', '1149.jpg', '1236.jpg', '1258.jpg', '1272.jpg', '1331.jpg', '1356.jpg', '210.jpg', '222.jpg', '3.jpg', '390.jpg', '431.jpg', '485.jpg', '505.jpg', '52.jpg', '559.jpg', '585.jpg', '634.jpg', '648.jpg', '670.jpg', '715.jpg', '784.jpg', '797.jpg', '803.jpg', '833.jpg', '848.jpg', '867.jpg', '899.jpg', '914.jpg', '940.jpg', '980.jpg', '993.jpg','1121.jpg', '1127.jpg', '1170.jpg', '1365.jpg', '321.jpg', '425.jpg', '542.jpg', '610.jpg', '896.jpg', '902.jpg', '953.jpg', '967.jpg']
    elif args.object == 'human':
        input_size = (608, 608)
        sampled_images = ['47.jpg','601.jpg','1304.jpg']
        skip_images = ['1022.jpg', '1041.jpg', '1053.jpg', '1063.jpg', '1066.jpg', '1097.jpg', '11.jpg', '1141.jpg', '1142.jpg', '1154.jpg', '1227.jpg', '1228.jpg', '1273.jpg', '1293.jpg', '1302.jpg', '1313.jpg', '1346.jpg', '1359.jpg', '1398.jpg', '1420.jpg', '1430.jpg', '1475.jpg', '1506.jpg', '152.jpg', '1538.jpg', '1553.jpg', '1624.jpg', '1663.jpg', '1664.jpg', '1746.jpg', '1770.jpg', '1788.jpg', '1803.jpg', '1805.jpg', '1817.jpg', '1852.jpg', '186.jpg', '1863.jpg', '1893.jpg', '19.jpg', '1917.jpg', '1954.jpg', '2008.jpg', '2040.jpg', '2087.jpg', '2092.jpg', '2108.jpg', '2121.jpg', '2128.jpg', '2141.jpg', '2161.jpg', '2186.jpg', '2203.jpg', '2219.jpg', '2226.jpg', '2262.jpg', '2270.jpg', '2271.jpg', '2279.jpg', '231.jpg', '2312.jpg', '2327.jpg', '2334.jpg', '2457.jpg', '250.jpg', '286.jpg', '348.jpg', '388.jpg', '391.jpg', '415.jpg', '422.jpg', '425.jpg', '452.jpg', '47.jpg', '640.jpg', '608.jpg', '670.jpg', '683.jpg', '748.jpg', '757.jpg', '805.jpg', '808.jpg', '829.jpg', '845.jpg', '85.jpg', '875.jpg', '897.jpg', '900.jpg', '928.jpg', '962.jpg', '97.jpg', '997.jpg']

    # default vehicle
    class_names_sel = ['car', 'bus', 'truck']
    # args.model_path = 'yolov5sbdd100k300epoch.pt'
    if args.object=='human':
        class_names_sel = ['person', 'rider']
    elif args.object=='COCO':
        class_names_sel = [line.strip() for line in open(args.coco_labels)] #FIXME: For specific category only

    if args.object == 'COCO':
        class_names_gt = [line.strip() for line in open(args.coco_labels)]
    else:        
        class_names_gt = ['person', 'rider', 'car', 'bus', 'truck']

        
    model = YOLOV5TorchObjectDetector(args.model_path, sel_nms, args.prob, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))

    # class_names_sel at this point: all possible categories in the experiment (defined in COCO_class.txt)
    saliency_method = YOLOV5XAI(model=model, excluded_layers=excluded_layers, sel_prob_str=args.prob,
                                    sel_norm_str=sel_norm, sel_classes=class_names_sel, sel_XAImethod=args.method,
                                    layers=target_layer_group_dict, sigma_factors=sigma_factors, img_size=input_size,)

    if os.path.isdir(args.img_path):
        img_list = os.listdir(args.img_path)
        img_list.sort()
        label_list = os.listdir(args.label_path)
        label_list.sort()
        # print(img_list)
        for item_img, item_label in zip(img_list[int(args.img_start):int(args.img_end)], label_list[int(args.img_start):int(args.img_end)]):

            if item_img in skip_images: continue # model failed to detect the target
            if item_img not in sampled_images: continue

            # if item_img not in ['34.jpg']: continue

            sigma_factors_to_run = sigma_factors.copy()

            # TODO: check and save only missing layers
            # for sigma_factor in sigma_factors:
            #     if sigma_factor == -1:
            #         if os.path.exists(os.path.join(args.output_dir.replace('gaussian_sigmaFACTOR','bilinear'), split_extension(item_img,suffix='-res'))) and\
            #             os.path.exists(os.path.join(args.output_dir.replace('gaussian_sigmaFACTOR','bilinear'), f"{split_extension(item_img,suffix='-res')}.mat")) and\
            #             os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','odam').replace('gaussian_sigmaFACTOR','bilinear'), split_extension(item_img,suffix='-res'))) and\
            #             os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','odam').replace('gaussian_sigmaFACTOR','bilinear'), f"{split_extension(item_img,suffix='-res')}.mat")) and\
            #             os.path.exists(os.path.join(args.output_dir.replace('gaussian_sigmaFACTOR','act_bilinear'), split_extension(item_img,suffix='-res'))) and\
            #             os.path.exists(os.path.join(args.output_dir.replace('gaussian_sigmaFACTOR','act_bilinear'), f"{split_extension(item_img,suffix='-res')}.mat")) and\
            #             os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','odam').replace('gaussian_sigmaFACTOR','act_bilinear'), split_extension(item_img,suffix='-res'))) and\
            #             os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','odam').replace('gaussian_sigmaFACTOR','act_bilinear'), f"{split_extension(item_img,suffix='-res')}.mat")):
            #             sigma_factors_to_run.remove(sigma_factor)
                
            #     elif os.path.exists(os.path.join(args.output_dir.replace('FACTOR',str(sigma_factor)), split_extension(item_img,suffix='-res'))) and\
            #         os.path.exists(os.path.join(args.output_dir.replace('FACTOR',str(sigma_factor)), f"{split_extension(item_img,suffix='-res')}.mat")) and\
            #         os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','odam').replace('FACTOR',str(sigma_factor)), split_extension(item_img,suffix='-res'))) and\
            #         os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','odam').replace('FACTOR',str(sigma_factor)), f"{split_extension(item_img,suffix='-res')}.mat")) and\
            #         os.path.exists(os.path.join(args.output_dir.replace("gaussian","act_gaussian").replace('FACTOR',str(sigma_factor)), split_extension(item_img,suffix='-res'))) and\
            #         os.path.exists(os.path.join(args.output_dir.replace("gaussian","act_gaussian").replace('FACTOR',str(sigma_factor)), f"{split_extension(item_img,suffix='-res')}.mat")) and\
            #         os.path.exists(os.path.join(args.output_dir.replace("gaussian","act_gaussian").replace('fullgradcamraw','odam').replace('FACTOR',str(sigma_factor)), split_extension(item_img,suffix='-res'))) and\
            #         os.path.exists(os.path.join(args.output_dir.replace("gaussian","act_gaussian").replace('fullgradcamraw','odam').replace('FACTOR',str(sigma_factor)), f"{split_extension(item_img,suffix='-res')}.mat")):
            #         sigma_factors_to_run.remove(sigma_factor)

            if len(sigma_factors_to_run)==0: continue

            if args.object == 'COCO':
                class_name = re.sub(r"_\d+\.(jpg|png)",'',item_img).replace('_',' ')
                if class_name not in class_names_gt:
                    print(f'[WARNING] {item_img} category parsed as {class_name}')
                    continue

                class_names_sel = [class_name]
                saliency_method.sel_classes = class_names_sel # generate saliency maps for specific category

            print(f"[INFO] Start processing {item_img}")
            failed = main(os.path.join(args.img_path, item_img), os.path.join(args.label_path, item_label), model, saliency_method, item_img[:-4],
                            class_names_sel,class_names_gt,
                            args,
                            sigma_factors=sigma_factors_to_run,
                            save_visualization=save_visualization)
            
            if failed:
                failed_imgs.append(item_img)

            # del model, saliency_method
            gc.collect()
            torch.cuda.empty_cache()
            # gpu_usage()

        else:
            main(args.img_path)
