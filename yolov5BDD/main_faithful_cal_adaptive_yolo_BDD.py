

import os
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
import math, logging
import utils.util_my_yolov5 as ut

import torch.utils.data
# from torch.utils.tensorboard import SummaryWriter
#
# import test  # import test.py to get mAP after each epoch
# from models.yolo import Model
# from utils import google_utils
from utils.datasets import *
from utils.utils import *

import pandas as pd

# Hyperparameter Settings
target_layer_group_dict = {
    "F1" : ['model_25_act', 'model_25_act', 'model_25_act'],
    "F2" : ['model_23_act', 'model_23_act', 'model_23_act'],
    "F3" : ['model_21_act', 'model_21_act', 'model_21_act'],
    "F4" : ['model_19_act', 'model_19_act', 'model_19_act'],
    "F5" : ['model_17_act', 'model_17_act', 'model_17_act'],
    "F6" : ['model_14_act', 'model_14_act', 'model_14_act'],
    "F7" : ['model_13_act', 'model_13_act', 'model_13_act'],
    "F8" : ['model_10_act', 'model_10_act', 'model_10_act'],
    "F9" : ['model_9_act', 'model_9_act', 'model_9_act'],
    "F10" : ['model_8_cv2_act', 'model_8_cv2_act', 'model_8_cv2_act'], 
    "F11" : ['model_7_act', 'model_7_act', 'model_7_act'], 
    "F12" : ['model_6_act', 'model_6_act', 'model_6_act'], 
    "F13" : ['model_5_act', 'model_5_act', 'model_5_act'], 
    "F14" : ['model_4_act', 'model_4_act', 'model_4_act'], 
    "F15" : ['model_3_act', 'model_3_act', 'model_3_act'], 
    "F16" : ['model_2_act', 'model_2_act', 'model_2_act'], 
    "F17" : ['model_1_act', 'model_1_act', 'model_1_act']
}

skipped_levels = {
    0 : ['F1','F2','F3','F4'],
    1 : ['F1','F2'],
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
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
parser.add_argument('--label-path', type=str, default=input_main_dir_label, help='input label path')

parser.add_argument('--object', type=str, default="vehicle", help='human or vehicle')
parser.add_argument('--prob', type=str, default="class", help='obj, class, objclass')
parser.add_argument('--coco-labels', type=str, default="COCO_classes.txt", help='path to coco classes list')

args = parser.parse_args()

# default vehicle
class_names_sel = ['car', 'bus', 'truck']
# args.model_path = 'yolov5sbdd100k300epoch.pt'
if args.object=='human':
    class_names_sel = ['person', 'rider']
    input_main_dir = 'orib_hum_id_task1009'
    input_main_dir_label = 'orib_hum_id_task1009_label'
elif args.object=='COCO':
    class_names_sel = [line.strip() for line in open(args.coco_labels)]
    # args.model_path = 'yolov5s_COCOPretrained.pt'
    input_main_dir = 'COCO_YOLO_IMAGE'
    input_main_dir_label = 'COCO_YOLO_LABEL'


gc.collect()
torch.cuda.empty_cache()
gpu_usage()

logging.basicConfig(filename="./odam.log",
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
    bb_selections = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/mscoco/other/for_eyegaze_GT_infos.xlsx')


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


def main(img_path, label_path, model, saliency_method, img_num,target_layer_group_name):
    gc.collect()
    torch.cuda.empty_cache()

    if args.object == 'COCO':
        class_names_gt = [line.strip() for line in open(args.coco_labels)]
    else:        
        class_names_gt = ['person', 'rider', 'car', 'bus', 'truck']

    img = cv2.imread(img_path)
    torch_img = model.preprocessing(img[..., ::-1])

    if args.object=='vehicle':
        bb_selection = bb_selections.loc[bb_selections['image']==img_path.split('/')[-1]] # 1029.jpg
    elif args.object=='human':
        bb_selection = bb_selections.loc[bb_selections['imgnumber']==int(img_path.split('/')[-1].replace('.jpg',''))] # 1029.jpg
    elif args.object=='COCO':
        bb_selection = bb_selections.loc[bb_selections['img']==img_path.split('/')[-1].replace('.jpg','')] # horse_382088.png

    tic = time.time()
    masks, masks_sum, [boxes, _, class_names, obj_prob], class_prob_list, head_num_list, raw_data = saliency_method(torch_img)
    print("total time:", round(time.time() - tic, 4))
    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr
    images = [result]

    ### New Images
    result_raw = result
    images = [img]
    result = img
    for i in range(len(masks)):
        masks[i] = F.upsample(masks[i], size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
    # masks[0] = masks[0]
    masks_sum = F.upsample(masks_sum, size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)

    ### Rescale Boxes
    shape_raw = [np.size(result_raw, 1), np.size(result_raw, 0)]  # w, h
    shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
    boxes_rescale_xyxy, boxes_rescale_xywh, boxes = ut.rescale_box_list(boxes, shape_raw, shape_new)

    ### Load labels
    boxes_GT, label_data_corr_xyxy, label_data_corr_xywh, label_data_corr_yxyx, label_data_class, label_data_class_names\
        = ut.load_gt_labels(img, label_path, class_names_gt, class_names_sel)

    ### Calculate AI Performance
    if len(boxes):
        Vacc = ut.calculate_acc(boxes_rescale_xywh, label_data_corr_xywh)/len(boxes_GT)
    else:
        Vacc = 0

    # Generalize to both whole-image and instance-based saliency maps
    if args.method == 'odam':
        # Find best matching target BB's index from pred (y1,x1,y2,x2)
        # For MSCOCO: Chenyang provided both GT target index and coordinates x1,y1,x2,y2
        if args.object == 'COCO':
            target_bbox_GT = bb_selection[['y1','x1','y2','x2']]
        # For BDD: we only know the target index. Get GT target BB coordiates from input annotations
        else:
            indices_GT_sorted = np.concatenate(boxes_GT,axis=0)[:, 1].argsort() # use x1 (top-left) to determine order of target (left to right)
            target_idx_GT = indices_GT_sorted[bb_selection['ExpTargetIndex'].values[0]-1]
            target_bbox_GT = boxes_GT[target_idx_GT][0]

        overlaps = np.zeros(len(boxes))
        target_indices_pred = [] # May miss the target
        for i in range(len(boxes)):
            overlaps[i] = area(boxes[i][0],target_bbox_GT)
        if len(overlaps) > 0 and overlaps.max() > 0:
            target_indices_pred.append(np.argmax(overlaps))
    else:
        # preserve all prediction boxes
        target_indices_pred = range(len(boxes))

    if len(target_indices_pred) > 0 and \
        target_layer_group_name in skipped_levels[int(head_num_list[target_indices_pred[0]][0])]:
        logging.info(f"{img_path} skip {args.target_layer}, head_num={int(head_num_list[target_indices_pred[0]][0])}")
        return

    # # Skip layers with empty saliency maps due to shorter pathways
    # if len(target_indices_pred) > 0 and np.isnan(masks[target_indices_pred[0]]).all().item() == 1:
    #     logging.info(f"{img_path} has empty saliency maps at {args.target_layer}, head_num={head_num_list[target_indices_pred[0]]}")
    #     return
    
    ### Display

    # Generate whole-image saliency maps as reference        
    all_mask_img = result.copy()
    all_mask_img, heat_map = ut.get_res_img(masks_sum, all_mask_img)

    res_img = result.copy()
    for i, mask in enumerate(masks):
        if i in target_indices_pred:
            res_img, heat_map = ut.get_res_img(mask, res_img)
        # else:
        #     res_img, heat_map = ut.get_res_img(torch.empty_like(mask), res_img)
        # FIXME: color scale when visualizing empty / zeros heatmaps where no maps found at target BB
    for i, (bbox, cls_name, obj_logit, class_prob, head_num) in enumerate(zip(boxes, class_names, obj_prob, class_prob_list, head_num_list)):
        if cls_name[0] in class_names_sel:
            if i in target_indices_pred:
                #bbox, cls_name = boxes[0][i], class_names[0][i]
                # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
                res_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob.cpu().detach().numpy()[0]*100)[:2] + ", " + str(head_num[0])[:1], res_img) / 255
                all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob.cpu().detach().numpy()[0]*100)[:2] + ", " + str(head_num[0])[:1], all_mask_img) / 255
            else:
                all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob.cpu().detach().numpy()[0]*100)[:2] + ", " + str(head_num[0])[:1], all_mask_img, color=(0,0,255)) / 255

    ## Display Ground Truth
    gt_img = result.copy()
    gt_img = gt_img / gt_img.max()
    for i, (bbox, cls_idx) in enumerate(zip(boxes_GT, label_data_class)):
        cls_idx = np.int8(cls_idx)
        if class_names_gt[cls_idx] in class_names_sel:
            if i==target_idx_GT:
                #bbox, cls_name = boxes[0][i], class_names[0][i]
                # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
                gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img,color=(0,255,0)) / 255
            else:
                gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img,color=(0,0,255)) / 255

    # images.append(gt_img * 255)
    images = [gt_img * 255]

    # no matching prediction and therefore empty saliency maps
    if len(target_indices_pred) == 0:
        images.append(res_img) # original image
    else:
        images.append(res_img * 255)
    images.append(all_mask_img * 255)
    final_image = ut.concat_images(images)
    img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
    output_path = f'{args.output_dir}/{img_name}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[INFO] Saving the final image at {output_path}')
    cv2.imwrite(output_path, final_image)

    gc.collect()
    torch.cuda.empty_cache()

    if len(target_indices_pred) == 0:
        masks_ndarray = np.zeros(masks[0].squeeze().detach().cpu().numpy().shape)
    else:
        masks_ndarray = masks[target_indices_pred[0]].squeeze().detach().cpu().numpy()

    # nofaith, aifaith, humanfaith, aihumanfaith
    if sel_faith == 'nofaith':
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                      'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                      'boxes_pred_xywh': boxes_rescale_xywh,
                                                      'boxes_gt_xywh': label_data_corr_xywh,
                                                      'boxes_gt_xyxy': label_data_corr_xyxy,
                                                      'HitRate': Vacc,
                                                      'boxes_pred_conf': obj_prob,
                                                      'boxes_pred_class_names': class_names,
                                                      'class_names_sel': class_names_sel,
                                                      'boxes_gt_classes_names': label_data_class_names,
                                                      'grad_act': raw_data
                                                      })
    elif sel_faith == 'aifaith':
        # AI Saliency Map Computation
        preds_deletion, preds_insertation, _ = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh, class_names_sel)
        # Saving
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                      'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                      'boxes_pred_xywh': boxes_rescale_xywh,
                                                      'boxes_gt_xywh': label_data_corr_xywh,
                                                      'boxes_gt_xyxy': label_data_corr_xyxy,
                                                      'HitRate': Vacc,
                                                      'preds_deletion': preds_deletion,
                                                      'preds_insertation': preds_insertation,
                                                      'boxes_pred_conf': obj_prob,
                                                      'boxes_pred_class_names': class_names,
                                                      'class_names_sel': class_names_sel,
                                                      'boxes_gt_classes_names': label_data_class_names,
                                                      'grad_act': raw_data
                                                      })
    elif sel_faith == 'humanfaith':
        # Human Saliency Map Loading
        human_saliency_map_path = '/mnt/D/HKU_XAI_Project/XAI_Human_compare_results_1/human_saliency_map_veh_new_1/' + img_num + '_GSmo_30.mat'
        human_saliency_map = scipy.io.loadmat(human_saliency_map_path)['output_map_norm']
        human_deletion, human_insertation, _ = ut.compute_faith(model, img, human_saliency_map, label_data_corr_xywh, class_names_sel)
        # Saving
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                      'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                      'boxes_pred_xywh': boxes_rescale_xywh,
                                                      'boxes_gt_xywh': label_data_corr_xywh,
                                                      'boxes_gt_xyxy': label_data_corr_xyxy,
                                                      'HitRate': Vacc,
                                                      'human_deletion': human_deletion,
                                                      'human_insertation': human_insertation,
                                                      'boxes_pred_conf': obj_prob,
                                                      'boxes_pred_class_names': class_names,
                                                      'class_names_sel': class_names_sel,
                                                      'boxes_gt_classes_names': label_data_class_names,
                                                      'grad_act': raw_data
                                                      })
    elif sel_faith == 'aihumanfaith':
        # AI Saliency Map Computation
        if args.object == 'COCO':
            preds_deletion, preds_insertation, _ = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh, class_names_sel)
        else:
            preds_deletion, preds_insertation, _ = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh)
        # Human Saliency Map Loading
        human_saliency_map_path = '/mnt/c/D/HKU_XAI_Project/XAI_Human_compare_results_1/human_saliency_map_1/' + img_num + '_GSmo_30.mat'
        human_saliency_map = scipy.io.loadmat(human_saliency_map_path)['output_map_norm']
        human_deletion, human_insertation, _ = ut.compute_faith(model, img, human_saliency_map, label_data_corr_xywh, class_names_sel)
        # Saving
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                      'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                      'boxes_pred_xywh': boxes_rescale_xywh,
                                                      'boxes_gt_xywh': label_data_corr_xywh,
                                                      'boxes_gt_xyxy': label_data_corr_xyxy,
                                                      'HitRate': Vacc,
                                                      'preds_deletion': preds_deletion,
                                                      'preds_insertation': preds_insertation,
                                                      'human_deletion': human_deletion,
                                                      'human_insertation': human_insertation,
                                                      'boxes_pred_conf': obj_prob,
                                                      'boxes_pred_class_names': class_names,
                                                      'class_names_sel': class_names_sel,
                                                      'boxes_gt_classes_names': label_data_class_names,
                                                      'grad_act': raw_data
                                                      })
    elif sel_faith == 'trainedXAIfaith':
        # Human Saliency Map Loading
        # trainedXAI_saliency_map_path = 'E:/HKU/HKU_XAI_Project/Human_Inspired_XAI_Try2/saveRawSaliencyMapData_testSet_GaussianConv/' + img_num + '_trainedSaliencyMap.mat'
        # trainedXAI_saliency_map_path = 'E:/HKU/HKU_XAI_Project/Human_Inspired_XAI_Try2/saveRawSaliencyMapData_testSet_GaussianConv_F10/' + img_num + '_trainedSaliencyMap.mat'
        trainedXAI_saliency_map_path = '/mnt/h/Projects/HKU_XAI_Project/Human_Inspired_XAI_Try2/saveRawSaliencyMapData_testSet_GaussianConv_correct/' + img_num + '_trainedSaliencyMap.mat'

        trainedXAI_saliency_map = scipy.io.loadmat(trainedXAI_saliency_map_path)['PredData_raw']
        trainedXAI_deletion, trainedXAI_insertation, _ = ut.compute_faith(model, img, trainedXAI_saliency_map, label_data_corr_xywh, class_names_sel)
        # Saving
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                      'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                      'boxes_pred_xywh': boxes_rescale_xywh,
                                                      'boxes_gt_xywh': label_data_corr_xywh,
                                                      'boxes_gt_xyxy': label_data_corr_xyxy,
                                                      'HitRate': Vacc,
                                                      'trainedXAI_deletion': trainedXAI_deletion,
                                                      'trainedXAI_insertation': trainedXAI_insertation,
                                                      'boxes_pred_conf': obj_prob,
                                                      'boxes_pred_class_names': class_names,
                                                      'class_names_sel': class_names_sel,
                                                      'boxes_gt_classes_names': label_data_class_names,
                                                      'grad_act': raw_data
                                                      })

    print(f'[INFO] save mat to: {output_path}')

if __name__ == '__main__':
    device = args.device
    # input_size = (args.img_size, args.img_size)
    input_size = (608, 608)

    print(f'[INFO] {args}')
    print('[INFO] Loading the model')

    model = YOLOV5TorchObjectDetector(args.model_path, sel_nms, args.prob, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))

    for i, (target_layer_group_name, target_layer_group) in enumerate(target_layer_group_dict.items()):
        if target_layer_group_name not in ["F1","F2","F3","F4","F5","F6","F7","F8"]: continue
        sub_dir_name = args.method + '_' + args.object + '_' + sel_nms + '_' + args.prob + '_' + target_layer_group_name + '_' + sel_faith + '_' + sel_norm + '_' + args.model_path.split('/')[-1][:-3] + '_' + '1'
        args.output_dir = os.path.join(args.output_main_dir, sub_dir_name)
        args.target_layer = target_layer_group
        saliency_method = YOLOV5XAI(model=model, layer_names=args.target_layer, sel_prob_str=args.prob,
                                        sel_norm_str=sel_norm, sel_classes=class_names_sel, sel_XAImethod=args.method, img_size=input_size)
        logging.info(target_layer_group)
        if os.path.isdir(args.img_path):
            img_list = os.listdir(args.img_path)
            label_list = os.listdir(args.label_path)
            # print(img_list)
            for item_img, item_label in zip(img_list, label_list):

                if os.path.exists(os.path.join(args.output_dir, split_extension(item_img,suffix='-res'))):
                    continue


                main(os.path.join(args.img_path, item_img), os.path.join(args.label_path, item_label), model, saliency_method, item_img[:-4],target_layer_group_name)

                # del model, saliency_method
                gc.collect()
                torch.cuda.empty_cache()
                gpu_usage()

        else:
            main(args.img_path)
