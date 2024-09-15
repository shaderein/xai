import warnings
warnings.filterwarnings("ignore")

import os,re
import time
import argparse
import numpy as np
from models.xai_method_odam import YOLOV5XAI
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

import pandas as pd

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hyperparameter Settings
target_layer_group_dict = {
    "model_0_act" : [6,2,2],
    
    "model_1_act": [3,2,1],
    "model_2" : {
        "model_2_cv1_act":[1,1,0],
        "model_2_cv2_act":[1,1,0]
    },
    "model_2_m_0_cv1_act":[1,1,0],
    "model_2_m_0_cv2_act":[3,1,1],
    "model_2_cv3_act":[1,1,0],

    "model_3_act": [3,2,1],
    "model_4" : {
        "model_4_cv1_act":[1,1,0],
        "model_4_cv2_act":[1,1,0]
    },
    "model_4_m_0_cv1_act":[1,1,0],
    "model_4_m_0_cv2_act":[3,1,1],
    "model_4_m_1_cv1_act":[1,1,0],
    "model_4_m_1_cv2_act":[3,1,1],
    "model_4_cv3_act":[1,1,0],

    "model_5_act": [3,2,1],
    "model_6" : {
        "model_6_cv1_act":[1,1,0],
        "model_6_cv2_act":[1,1,0]
    },
    "model_6_m_0_cv1_act":[1,1,0],
    "model_6_m_0_cv2_act":[3,1,1],
    "model_6_m_1_cv1_act":[1,1,0],
    "model_6_m_1_cv2_act":[3,1,1],
    "model_6_m_2_cv1_act":[1,1,0],
    "model_6_m_2_cv2_act":[3,1,1],
    "model_6_cv3_act":[1,1,0],

    "model_7_act": [3,2,1],
    "model_8" : {
        "model_8_cv1_act":[1,1,0],
        "model_8_cv2_act":[1,1,0]
    },
    "model_8_m_0_cv1_act":[1,1,0],
    "model_8_m_0_cv2_act":[3,1,1],
    "model_8_cv3_act":[1,1,0],

    # SPPF
    "model_9_cv1_act":[1,1,0],
    "model_9_m_act1":[5,1,2], # repeated
    "model_9_m_act2":[5,1,2],
    "model_9_m_act3":[5,1,2],
    "model_9_cv2_act":[1,1,0],

    "model_10_act":[1,1,0],
    "model_13" : {
        "model_13_cv1_act":[1,1,0],
        "model_13_cv2_act":[1,1,0]
    },
    "model_13_m_0_cv1_act":[1,1,0],
    "model_13_m_0_cv2_act":[3,1,1],
    "model_13_cv3_act":[1,1,0],

    "model_14_act":[1,1,0],
    "model_17" : {
        "model_17_cv1_act":[1,1,0],
        "model_17_cv2_act":[1,1,0]
    },
    "model_17_m_0_cv1_act":[1,1,0],
    "model_17_m_0_cv2_act":[3,1,1],
    "model_17_cv3_act":[1,1,0],

    "model_18_act":[3,2,1],
    "model_20" : {
        "model_20_cv1_act":[1,1,0],
        "model_20_cv2_act":[1,1,0]
    },
    "model_20_m_0_cv1_act":[1,1,0],
    "model_20_m_0_cv2_act":[3,1,1],
    "model_20_cv3_act":[1,1,0],

    "model_21_act":[3,2,1],
    "model_23" : {
        "model_23_cv1_act":[1,1,0],
        "model_23_cv2_act":[1,1,0]
    },
    "model_23_m_0_cv1_act":[1,1,0],
    "model_23_m_0_cv2_act":[3,1,1],
    "model_23_cv3_act":[1,1,0],    
}

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
parser.add_argument('--img-size', type=int, default=640, help="input image size")
# parser.add_argument('--target-layer', type=list, default=list(target_layer_group_list[0]),
#                     help='The layer hierarchical address to which gradcam will applied,'
#                          ' the names should be separated by underline')

parser.add_argument('--method', type=str, default="gradcam", help='gradcam or eigencam or eigengradcam or weightedgradcam or gradcampp or fullgradcam')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
parser.add_argument('--label-path', type=str, default=input_main_dir_label, help='input label path')

parser.add_argument('--object', type=str, default="vehicle", help='human or vehicle')
parser.add_argument('--prob', type=str, default="class", help='obj, class, objclass')
parser.add_argument('--coco-labels', type=str, default="COCO_classes.txt", help='path to coco classes list')

args = parser.parse_args()

args.object = "COCO"
args.model_path = "/mnt/h/jinhan/xai/models/yolov5s_COCOPretrained.pt"
args.method = "fullgradcamraw"
args.prob = "class"
args.output_main_dir = "/mnt/h/jinhan/results/mscoco/xai_saliency_maps_yolov5s_gaussian_sigmaFACTOR/fullgradcamraw"
args.coco_labels = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/COCO_classes2.txt"
args.img_path = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/DET2"
args.label_path = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/annotations_DET2"

# default vehicle
class_names_sel = ['car', 'bus', 'truck']
# args.model_path = 'yolov5sbdd100k300epoch.pt'
if args.object=='human':
    class_names_sel = ['person', 'rider']
    input_main_dir = 'orib_hum_id_task1009'
    input_main_dir_label = 'orib_hum_id_task1009_label'
elif args.object=='COCO':
    class_names_sel = [line.strip() for line in open(args.coco_labels)] #FIXME: For specific category only
    # args.model_path = 'yolov5s_COCOPretrained.pt'
    input_main_dir = 'COCO_YOLO_IMAGE'
    input_main_dir_label = 'COCO_YOLO_LABEL'

if args.object == 'COCO':
    class_names_gt = [line.strip() for line in open(args.coco_labels)]
else:        
    class_names_gt = ['person', 'rider', 'car', 'bus', 'truck']



gc.collect()
torch.cuda.empty_cache()
gpu_usage()

logging.basicConfig(filename="./whole_image.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# Select BB used in EXP
if args.object=='human':
    bb_selections = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/bdd/labels_mapping/Random_sample_human_procedure_analysis.xlsx','hum_sample_img_condition')
    bb_selections = bb_selections[['imgnumber','human_count_gt','ExpTargetIndex']]
elif args.object=='vehicle':
    bb_selections = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/bdd/labels_mapping/Random_sample_vehicle_procedure_analysis.xlsx','veh_sample_img_condition')
    bb_selections = bb_selections[['image','vehicle_count_gt','ExpTargetIndex']]
# NOTE: From Chenyang. Check his data format.
elif args.object=='COCO':
    bb_selections = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/mscoco/other/for_eyegaze_GT_infos_size_ratio.xlsx')


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

def main(img_path, label_path, model, saliency_method, img_num, class_names_sel,target_layer_group_name,
         sigma_factors=[-1,2,4]):
    gc.collect()
    torch.cuda.empty_cache()

    img = cv2.imread(img_path)
    torch_img = model.preprocessing(img[..., ::-1])

    # Find instance used in experiments
    bb_selections = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/mscoco/other/for_eyegaze_GT_infos_size_ratio.xlsx')
    bb_selection = bb_selections.loc[bb_selections['img']==img_path.split('/')[-1].replace('.jpg','')] # horse_382088.png

    tic = time.time()

    masks_all, masks_sum_all, [boxes, _, class_names, obj_prob], class_prob_list, head_num_list, raw_data = saliency_method(torch_img,(img.shape[0],img.shape[1]),sigma_factors)
    print("[Raw] total time:", round(time.time() - tic, 4))

    if len(boxes) == 0: 
        print(f"Failed to predict {img_path}")
        return

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
    target_bbox_GT = [bb_selection[['y1']].y1.item()*img.shape[0],
                bb_selection[['x1']].x1.item()*img.shape[1],
                bb_selection[['y2']].y2.item()*img.shape[0],
                bb_selection[['x2']].x2.item()*img.shape[1]]
    boxes_GT_overlaps = [bbox_iou(box_GT[0],target_bbox_GT) for box_GT in boxes_GT]
    target_idx_GT = np.argmax(boxes_GT_overlaps)

    overlaps = np.zeros(len(boxes))
    target_indices_pred = [] # May miss the target
    for i in range(len(boxes)):
        # overlaps[i] = area(boxes[i][0],target_bbox_GT,threshold=0.1)
        overlaps[i] = bbox_iou(boxes[i][0],target_bbox_GT)
        if overlaps[i] < 0.1: overlaps[i] = 0
    if len(overlaps) > 0 and overlaps.max() > 0:
        target_indices_pred.append(np.argmax(overlaps))
    
    for sigma_factor in masks_all:

        masks = masks_all[sigma_factor]
        masks_sum = masks_sum_all[sigma_factor]

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

        ### Display
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
        if sigma_factor == -1:
            output_path = f'{args.output_dir.replace("gaussian_sigmaFACTOR","bilinear")}/{img_name}'
            os.makedirs(args.output_dir.replace("gaussian_sigmaFACTOR", "bilinear"), exist_ok=True)
        else:
            output_path = f'{args.output_dir.replace("FACTOR",str(sigma_factor))}/{img_name}'
            os.makedirs(args.output_dir.replace("FACTOR",str(sigma_factor)), exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        cv2.imwrite(output_path, final_image)

        gc.collect()
        torch.cuda.empty_cache()

        masks_ndarray = masks[0].squeeze().detach().cpu().numpy()

        start = time.time()
        # AI Saliency Map Computation
        preds_deletion, preds_insertation, _ = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh, class_names_sel)
        # Saving
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                    'masks_ndarray_all': [mask.squeeze().detach().cpu().numpy() for mask in masks_orig],
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

        ### Display

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
        images = [cv2.imread(f"/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/EXP/{img_path.split('/')[-1]}")]
        images.append(gt_img * 255)

        # no matching prediction and therefore empty saliency maps
        if len(target_indices_pred) == 0:
            images.append(res_img) # original image
        else:
            images.append(res_img * 255)

        images.append(all_mask_img * 255)
        final_image = ut.concat_images(images)

        img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
        odam_output_dir = args.output_dir.replace('fullgradcamraw','odam')
        if sigma_factor == -1:
            output_path = f'{odam_output_dir.replace("gaussian_sigmaFACTOR","bilinear")}/{img_name}'
            os.makedirs(odam_output_dir.replace("gaussian_sigmaFACTOR", "bilinear"), exist_ok=True)
        else:
            output_path = f'{odam_output_dir.replace("FACTOR",str(sigma_factor))}/{img_name}'
            os.makedirs(odam_output_dir.replace("FACTOR",str(sigma_factor)), exist_ok=True)
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
        preds_deletion, preds_insertation, _ = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh[[target_idx_GT]], class_names_sel)
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

if __name__ == '__main__':
    device = args.device
    # input_size = (args.img_size, args.img_size)
    input_size = (640, 640)
    sigma_factors = [-1,2,4]

    print(f'[INFO] {args}')
    print('[INFO] Loading the model')

    args.object = "COCO"
    args.model_path = "/mnt/h/jinhan/xai/models/yolov5s_COCOPretrained.pt"
    args.method = "fullgradcamraw"
    args.prob = "class"
    args.output_main_dir = "/mnt/h/jinhan/results/mscoco/xai_saliency_maps_yolov5s_gaussian_sigmaFACTOR/fullgradcamraw"
    args.coco_labels = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/COCO_classes2.txt"
    args.img_path = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/DET2"
    args.label_path = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/annotations_DET2"

    model = YOLOV5TorchObjectDetector(args.model_path, sel_nms, args.prob, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))

    flatten_layers = {}
    for name, param in target_layer_group_dict.items():
        if isinstance(param,dict):
            for n,p in param.items():
                flatten_layers[n] = p
        else:
            flatten_layers[name] = param

    for i, (target_layer_group_name,layer_param) in enumerate(flatten_layers.items()):
        if target_layer_group_name in ["model_0_act","model_9_m_act1","model_9_m_act2","model_9_m_act3"]: continue
        
        sub_dir_name = args.method + '_' + args.object + '_' + sel_nms + '_' + args.prob + '_' + target_layer_group_name + '_' + sel_faith + '_' + sel_norm + '_' + args.model_path.split('/')[-1][:-3] + '_' + '1'
        args.output_dir = os.path.join(args.output_main_dir, sub_dir_name)
        args.target_layer = [target_layer_group_name,target_layer_group_name,target_layer_group_name]

        # class_names_sel at this point: all possible categories in the experiment (defined in COCO_class.txt)
        saliency_method = YOLOV5XAI(model=model, layer_names=args.target_layer, sel_prob_str=args.prob,
                                        sel_norm_str=sel_norm, sel_classes=class_names_sel, sel_XAImethod=args.method,
                                        layers=target_layer_group_dict, sigma_factors=sigma_factors, img_size=input_size,)

        if os.path.isdir(args.img_path):
            img_list = os.listdir(args.img_path)
            label_list = os.listdir(args.label_path)
            # print(img_list)
            for item_img, item_label in zip(img_list, label_list):
                
                sampled_images = ['chair_81061.png','elephant_97230.png','giraffe_287545.png']
                skip_images = ["book_472678.png","potted plant_473219.png","bench_350607.png","truck_295420.png","toaster_232348.png","kite_405279.png","toothbrush_218439.png","snowboard_425906.png","car_227511.png","traffic light_453841.png","hair drier_239041.png","hair drier_178028.png","toaster_453302.png","mouse_513688.png","spoon_88040.png","scissors_340930.png","handbag_383842.png"]

                if item_img in skip_images: continue # model failed to detect the target
                if item_img not in sampled_images: continue

                sigma_factors_to_run = sigma_factors.copy()

                for sigma_factor in sigma_factors:
                    if sigma_factor == -1:
                        if os.path.exists(os.path.join(args.output_dir.replace('gaussian_sigmaFACTOR','bilinear'), split_extension(item_img,suffix='-res'))) and\
                            os.path.exists(os.path.join(args.output_dir.replace('gaussian_sigmaFACTOR','bilinear'), f"{split_extension(item_img,suffix='-res')}.mat")) and\
                            os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','odam').replace('gaussian_sigmaFACTOR','bilinear'), split_extension(item_img,suffix='-res'))) and\
                            os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','odam').replace('gaussian_sigmaFACTOR','bilinear'), f"{split_extension(item_img,suffix='-res')}.mat")):
                            sigma_factors_to_run.remove(sigma_factor)
                    elif os.path.exists(os.path.join(args.output_dir.replace('FACTOR',str(sigma_factor)), split_extension(item_img,suffix='-res'))) and\
                        os.path.exists(os.path.join(args.output_dir.replace('FACTOR',str(sigma_factor)), f"{split_extension(item_img,suffix='-res')}.mat")) and\
                        os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','odam').replace('FACTOR',str(sigma_factor)), split_extension(item_img,suffix='-res'))) and\
                        os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','odam').replace('FACTOR',str(sigma_factor)), f"{split_extension(item_img,suffix='-res')}.mat")):
                        sigma_factors_to_run.remove(sigma_factor)

                if len(sigma_factors_to_run)==0: continue

                class_name = re.sub(r"_\d+\.(jpg|png)",'',item_img).replace('_',' ')
                if class_name not in class_names_gt:
                    print(f'[WARNING] {item_img} category parsed as {class_name}')
                    continue

                saliency_method.sel_classes = [class_name] # generate saliency maps for specific category

                # try:
                main(os.path.join(args.img_path, item_img), os.path.join(args.label_path, item_label), model, saliency_method, item_img[:-4],[class_name],target_layer_group_name,
                    sigma_factors_to_run)
                # except RuntimeError as e:
                #     if 'CUDA' in str(e):
                #         print("Caught CUDA out of memory error!")
                #         raise e
                #     else: pass


                # del model, saliency_method
                gc.collect()
                torch.cuda.empty_cache()
                # gpu_usage()

        else:
            main(args.img_path)
