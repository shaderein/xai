import warnings
warnings.filterwarnings("ignore")

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os,re, json, sys

import cv2
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from grad_cam_head_faithfulness_rpn import GradCAM, apply_gaussian_kernel
from skimage import io
from torch import nn
from utils_previous import get_res_img, put_text_box, concat_images, calculate_acc, scale_coords_new, xyxy2xywh, xywh2xyxy
import util_my_yolov5 as ut

import argparse
from deep_utils import Box, split_extension
import scipy.io
import imageio
import pandas as pd
# from numba import cuda
from GPUtil import showUtilization as gpu_usage
import gc
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from matplotlib.image import imread
import math, time
from collections import defaultdict

import configparser
path_config = configparser.ConfigParser()
path_config.read('./config_rpn_bdd2coco.ini')

import logging
logging.basicConfig(filename=f"{path_config.get('Paths','log_dir')}/{path_config.get('Paths','log_file')}",
                    filemode='a',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# from torch.utils.tensorboard import SummaryWriter

# constants
WINDOW_NAME = "COCO detections"

# Hyperparameter Settings
# kernel, stride, padding
target_layer_group_dict = {
    'backbone.stem.conv1' : [7,2,3],
    'stem.MaxPool' : [3,2,1],
    'backbone.res2.0.conv1' : [1,1,0],
    'backbone.res2.0.conv2' : [3,1,1],
    'backbone.res2.0.conv3' : [1,1,0],
    'backbone.res2.1.conv1' : [1,1,0],
    'backbone.res2.1.conv2' : [3,1,1],
    'backbone.res2.1.conv3' : [1,1,0],
    'backbone.res2.2.conv1' : [1,1,0],
    'backbone.res2.2.conv2' : [3,1,1],
    'backbone.res2.2.conv3' : [1,1,0],
    'backbone.res3.0.conv1' : [1,2,0],
    'backbone.res3.0.conv2' : [3,1,1],
    'backbone.res3.0.conv3' : [1,1,0],
    'backbone.res3.1.conv1' : [1,1,0],
    'backbone.res3.1.conv2' : [3,1,1],
    'backbone.res3.1.conv3' : [1,1,0],
    'backbone.res3.2.conv1' : [1,1,0],
    'backbone.res3.2.conv2' : [3,1,1], 
    'backbone.res3.2.conv3' : [1,1,0],
    'backbone.res3.3.conv1' : [1,1,0],
    'backbone.res3.3.conv2' : [3,1,1],  
    'backbone.res3.3.conv3' : [1,1,0],
    'backbone.res4.0.conv1' : [1,2,0],
    'backbone.res4.0.conv2' : [3,1,1],  
    'backbone.res4.0.conv3' : [1,1,0],
    'backbone.res4.1.conv1' : [1,1,0],
    'backbone.res4.1.conv2' : [3,1,1],  
    'backbone.res4.1.conv3' : [1,1,0],
    'backbone.res4.2.conv1' : [1,1,0],
    'backbone.res4.2.conv2' : [3,1,1],  
    'backbone.res4.2.conv3' : [1,1,0],
    'backbone.res4.3.conv1' : [1,1,0],
    'backbone.res4.3.conv2' : [3,1,1],  
    'backbone.res4.3.conv3' : [1,1,0],
    'backbone.res4.4.conv1' : [1,1,0],
    'backbone.res4.4.conv2' : [3,1,1],  
    'backbone.res4.4.conv3' : [1,1,0],
    'backbone.res4.5.conv1' : [1,1,0],
    'backbone.res4.5.conv2' : [3,1,1], 
    'backbone.res4.5.conv3' : [1,1,0],
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
parser.add_argument('--model-path', type=str, default='', help='Path to the model')
parser.add_argument('--img-path', type=str, default=input_main_dir, help='input image path')
parser.add_argument('--output-dir', type=str, default='sample_EM_idtask_1_output_update_2/GradCAM_NMS_objclass_F0_singleScale_norm_v5s_1', help='output dir')
parser.add_argument('--output-main-dir', type=str, default=output_main_dir, help='output root dir')
parser.add_argument('--img-size', type=int, default=608, help="input image size")
# parser.add_argument('--target-layer', type=list, default='F1',
#                     help='The layer hierarchical address to which gradcam will applied,'
#                          ' the names should be separated by underline')

parser.add_argument('--method', type=str, default="fullgradcamraw", help='gradcam or eigencam or eigengradcam or weightedgradcam or gradcampp or fullgradcam')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
parser.add_argument('--label-path', type=str, default=input_main_dir_label, help='input label path')
parser.add_argument('--norm', type=str, default="norm")

parser.add_argument('--object', type=str, default="COCO", help='COCO, human or vehicle')
parser.add_argument('--prob', type=str, default="class", help='obj, class, objclass')
parser.add_argument('--coco-labels', type=str, default="COCO_classes.txt", help='path to coco classes list')

parser.add_argument('--img-start', type=int, default=0)
parser.add_argument('--img-end', type=int, default=160)

parser.add_argument('--layer-start', type=int, default=0)
parser.add_argument('--layer-end', type=int, default=51)

parser.add_argument('--visualize', action='store_true')

gc.collect()
torch.cuda.empty_cache()
gpu_usage()


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.freeze()
    return cfg


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """

        :param module:
        :param grad_in: tuple,长度为1
        :param grad_out: tuple,长度为1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        score = output[0]['instances'].scores[index]
        score.backward()

        return inputs['image'].grad  # [3,H,W]


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), heatmap


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network='frcnn', output_dir='./results'):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def get_parser(img_path, run_device, model_path):
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="./faster_rcnn_R_50_C4_1x.yaml", #"configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input",
                        default=img_path,
                        help="img_path")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", model_path, "MODEL.DEVICE", run_device],
        nargs=argparse.REMAINDER,
    )
    return parser

def compute_faith(model, img, masks_ndarray, label_data_class, cfg):
    height, width = img.shape[:2]
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image = transform_gen.get_transform(img).apply_image(img)

    valid_area = int(height * width)

    # Jinhan: DEBUG
    imgs_deletion = []
    imgs_insertation = []

    # torch_img = model.preprocessing(img[..., ::-1])

    ### Deletion
    # Sort Saliency Map
    delta_thr = 0
    sample_step = 10 # step number (delete/insert total/num_thr pixels each time)
    pixel_once = max(1, int(valid_area/sample_step))
    grids = ut.make_grids(height,width)

    masks_ndarray[np.isnan(masks_ndarray)] = 0
    masks_ndarray[masks_ndarray <= delta_thr] = 0
    if sum(sum(masks_ndarray)) == 0:
        masks_ndarray[0, 0] = 1
        masks_ndarray[1, 1] = 0.5
        return [],[],[],[],[]
    
    masks_ndarray_flatten = masks_ndarray.flatten()

    masks_ndarray_sort_idx = np.argsort(masks_ndarray_flatten)[::-1]  # descend

    masks_ndarray_RGB = np.expand_dims(masks_ndarray, 2)
    masks_ndarray_RGB = np.concatenate((masks_ndarray_RGB, masks_ndarray_RGB, masks_ndarray_RGB),2)

    img_raw = img
    img_raw_float = img_raw.astype('float')/255
    preds_deletion, proposals_deletion, proposals_inds_deletion = [], [], []    
    # Keep inserting on the same image
    img_raw_float_use = img_raw_float.copy()

    with torch.no_grad():
        for i in range(sample_step):
            perturb_grids = list(masks_ndarray_sort_idx[i*pixel_once:(i+1)*pixel_once])
            perturb_pos = grids[perturb_grids]
            perturb_x = perturb_pos[:, 0]
            perturb_y = perturb_pos[:, 1]
            img_raw_float_use[perturb_y,perturb_x,:] = np.random.rand(pixel_once,3)
            img_raw_uint8_use = (img_raw_float_use*255).astype('uint8')

            # imgs_deletion.append(img_raw_uint8_use[..., ::-1]) # Jinhan: save image to view deletion process

            # torch_img_rand = model.preprocessing(img_raw_uint8_use[..., ::-1])
            image = transform_gen.get_transform(img_raw_uint8_use).apply_image(img_raw_uint8_use)
            torch_img_rand = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": torch_img_rand, "height": height, "width": width}
            preds_deletion_i, proposals_i, proposal_inds_i = model.inference([inputs],do_postprocess=True, output_inds=True, output_proposals=True)
            preds_deletion.extend(preds_deletion_i)
            proposals_deletion.extend(proposals_i)
            proposals_inds_deletion.extend(proposal_inds_i)

    shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
    # Jinhan: unlike Yolov5, FasterRCNN doesn't have head indices in its output, but keep the return value ds consistent
    #   for the downstream faithfulness mAUC calculation
    pred_deletion_adj = [[[] for _ in range(sample_step)] for _ in range(5)]
    for i, (preds_deletion_i,proposals_i, proposal_inds_i) in enumerate(zip(preds_deletion,proposals_deletion,proposals_inds_deletion)):
        for bbox_one, cls_idx_one, proposal_idx in zip(preds_deletion_i['instances'].pred_boxes.tensor, preds_deletion_i['instances'].pred_classes, proposal_inds_i):
            if cls_idx_one.item() in label_data_class.astype(np.int64):

                conf_one = torch.sigmoid(proposals_i.objectness_logits[proposal_idx])
                
                # Jinhan: DEBUG
                # imgs_deletion[i] = put_text_box(bbox_one,'test',imgs_deletion[i]).astype('uint8')[...,[1,0,2]]
                
                boxes_rescale_xyxy, boxes_rescale_xywh, _ = rescale_box_list([[bbox_one.detach().cpu().numpy()[[1,0,3,2]]]], shape_new, shape_new) # yxyx
                pred_deletion_adj[0][i].append(boxes_rescale_xyxy.tolist()[0])
                pred_deletion_adj[1][i].append(boxes_rescale_xywh.tolist()[0])
                pred_deletion_adj[2][i].append(cls_idx_one.detach().cpu().numpy())
                pred_deletion_adj[4][i].append(conf_one.detach().cpu().numpy())

    ### Insertation
    preds_insertation, proposals_insertation, proposals_inds_insertation = [], [], []    
    with torch.no_grad():
        for i in range(sample_step):
            insert_grids = list(masks_ndarray_sort_idx[i*pixel_once:(i+1)*pixel_once])
            insert_pos = grids[insert_grids]
            insert_x = insert_pos[:, 0]
            insert_y = insert_pos[:, 1]

            img_raw_float_use[insert_y,insert_x,:] = img_raw_float[insert_y,insert_x,:]
            img_raw_uint8_use = (img_raw_float_use*255).astype('uint8')

            # Jinhan : DEBUG
            # imgs_insertation.append(img_raw_uint8_use[..., ::-1]) # Jinhan: save image to view insertion process

            # torch_img_rand = model.preprocessing(img_raw_uint8_use[..., ::-1])
            image = transform_gen.get_transform(img_raw_uint8_use).apply_image(img_raw_uint8_use)
            torch_img_rand = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": torch_img_rand, "height": height, "width": width}
            preds_insertation_i, proposals_i, proposal_inds_i = model.inference([inputs],do_postprocess=True, output_inds=True, output_proposals=True)
            preds_insertation.extend(preds_insertation_i)
            proposals_insertation.extend(proposals_i)
            proposals_inds_insertation.extend(proposal_inds_i)

    pred_insertation_adj = [[[] for _ in range(sample_step)] for _ in range(5)]
    for i, (preds_insertation_i,proposals_i, proposal_inds_i) in enumerate(zip(preds_insertation,proposals_insertation,proposals_inds_insertation)):
        for bbox_one, cls_idx_one, proposal_idx in zip(preds_insertation_i['instances'].pred_boxes.tensor, preds_insertation_i['instances'].pred_classes, proposal_inds_i):
            if cls_idx_one.item() in label_data_class.astype(np.int64):

                conf_one = torch.sigmoid(proposals_i.objectness_logits[proposal_idx])
                
                # Jinhan: DEBUG
                # imgs_insertation[i] = put_text_box(bbox_one,'test',imgs_insertation[i]).astype('uint8')[...,[1,0,2]]
                
                boxes_rescale_xyxy, boxes_rescale_xywh, _ = rescale_box_list([[bbox_one.detach().cpu().numpy()[[1,0,3,2]]]], shape_new, shape_new) # yxyx
                pred_insertation_adj[0][i].append(boxes_rescale_xyxy.tolist()[0])
                pred_insertation_adj[1][i].append(boxes_rescale_xywh.tolist()[0])
                pred_insertation_adj[2][i].append(cls_idx_one.detach().cpu().numpy())
                pred_insertation_adj[4][i].append(conf_one.detach().cpu().numpy())

    return pred_deletion_adj, pred_insertation_adj, None, imgs_deletion, imgs_insertation

def rescale_box_list(boxes, shape_raw, shape_new):
    if len(boxes):
        boxes_ndarray = np.array(boxes).squeeze(1)
        boxes_ndarray = torch.from_numpy(boxes_ndarray) #yxyx
        # img1_shape = [np.size(result_raw, 1), np.size(result_raw, 0)] #w, h
        # img0_shape = [np.size(img, 1), np.size(img, 0)] #w, h
        boxes_rescale = scale_coords_new(shape_raw, boxes_ndarray.float(), shape_new)
        boxes_rescale = boxes_rescale.round()
        boxes_rescale = torch.unsqueeze(boxes_rescale, 1)
        boxes = boxes_rescale.tolist()
        boxes_rescale_yxyx = boxes_rescale.squeeze(1).numpy()
        boxes_rescale_xyxy = boxes_rescale_yxyx[:, [1,0,3,2]]
        boxes_rescale_xywh = xyxy2xywh(boxes_rescale_xyxy)
    else:
        boxes_rescale_xyxy = 0
        boxes_rescale_xywh = 0

    return boxes_rescale_xyxy, boxes_rescale_xywh, boxes


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

def mean_valid_confidence(targets_corr, preds_corr_list, preds_conf_list, threshold=0.5):
    """
    corr: xywh
    """
    confidence_all_steps = np.zeros(len(preds_conf_list))

    for step, preds_corr in enumerate(preds_corr_list):
        if np.asarray(preds_corr).size == 0: 
            confidence_all_steps[step] = 0 # failed to predict

        valid_confidence = np.zeros(targets_corr.shape[0])

        for i, target_corr in enumerate(targets_corr):
            max_iou = 0
            for j, pred_corr in enumerate(preds_corr):
                iou = bbox_iou(target_corr, pred_corr)
                if iou > threshold and iou > max_iou:
                    valid_confidence[i] = preds_conf_list[step][j]
        confidence_all_steps[step] = valid_confidence.mean()

    return confidence_all_steps.mean()

def rescale_and_apply_gaussian(saliency_maps_orig_all, mapped_locs_all, h_orig, w_orig, sigma, sel_norm_str):
    masks = []
    masks_sum = torch.zeros((1,1,h_orig,w_orig))
    nObj = 0
    for saliency_map_orig, mapped_locs in zip(saliency_maps_orig_all, mapped_locs_all):
        if saliency_map_orig.max().item() != 0:
            nObj = nObj + 1

        if saliency_map_orig.max().item() == 0:
            saliency_map = torch.zeros((1,1,h_orig,w_orig))
        else:                            
            saliency_map = apply_gaussian_kernel(saliency_map_orig, mapped_locs, sigma, (h_orig,w_orig)).sum(0,keepdim=True)    

        if sel_norm_str == 'norm' and saliency_map_orig.max().item() != 0:
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        masks_sum = masks_sum + saliency_map
        masks.append(saliency_map)

    if nObj > 0:
        masks_sum = masks_sum / nObj

    return masks, masks_sum

def main(img_path, label_path, target_layer_group, 
         model, saliency_method, 
         arguments, cfg, args,
        class_names_gt, class_names_sel,
        save_visualization=False):
    img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')

    # Find instance used in experiments
    if args.object=='vehicle':
        bb_selections = pd.read_excel(f'{path_config.get("Paths", "data_dir")}/bdd/labels_mapping/Random_sample_vehicle_procedure_analysis.xlsx','veh_sample_img_condition')
        bb_selections = bb_selections[['image','vehicle_count_gt','ExpTargetIndex']]
        bb_selection = bb_selections.loc[bb_selections['image']==img_path.split('/')[-1]] # 1029.jpg
    elif args.object=='human':
        bb_selections = pd.read_excel(f'{path_config.get("Paths", "data_dir")}/bdd/labels_mapping/Random_sample_human_procedure_analysis.xlsx','hum_sample_img_condition')
        bb_selections = bb_selections[['imgnumber','human_count_gt','ExpTargetIndex']]
        bb_selection = bb_selections.loc[bb_selections['imgnumber']==int(img_path.split('/')[-1].replace('.jpg',''))] # 1029.jpg
    elif args.object == 'COCO':
        # Find instance used in experiments
        bb_selections = pd.read_excel(f'{path_config.get("Paths", "data_dir")}/mscoco/other/for_eyegaze_GT_infos_size_ratio.xlsx')
        bb_selection = bb_selections.loc[bb_selections['img']==img_path.split('/')[-1].replace('.jpg','')] # horse_382088.png

    start = time.time()

    # 加载图像
    img = read_image(img_path, format="BGR")
    height, width = img.shape[:2]
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image = transform_gen.get_transform(img).apply_image(img)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)

    inputs = {"image": image, "height": height, "width": width}

    masks_orig_all, mapped_locs, adjusted_receptive_field, proposals_all, [boxes, _, class_names], class_prob_list, raw_data = saliency_method(inputs)  # cam mask
    
    saliencyMap_method.remove_handlers()

    saliency_maps_orig_all, activation_maps_orig_all = masks_orig_all

    raw_masks = {
        "saliency_maps_orig_all" : saliency_maps_orig_all,
        "activation_maps_orig_all" : activation_maps_orig_all,
    }
    os.makedirs(args.output_dir.replace('fullgradcamraw','raw_masks'), exist_ok=True)
    torch.save(raw_masks,os.path.join(args.output_dir.replace('fullgradcamraw','raw_masks'),f"{img_path.split('/')[-1].split('.')[0]}.pth"))

    torch_img = image.unsqueeze(0)
    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr

    ### New Images
    result_raw = result
    result = img

    ### Rescale Boxes
    if len(boxes):
        boxes_ndarray = np.array(boxes).squeeze(1)
        boxes_ndarray = torch.from_numpy(boxes_ndarray) #yxyx
        # img1_shape = [np.size(result_raw, 1), np.size(result_raw, 0)] #w, h
        img1_shape = [np.size(img, 1), np.size(img, 0)] #w, h
        img0_shape = [np.size(img, 1), np.size(img, 0)] #w, h
        boxes_rescale = scale_coords_new(img1_shape, boxes_ndarray.float(), img0_shape, ratio_pad=None)
        boxes_rescale = boxes_rescale.round()
        boxes_rescale = torch.unsqueeze(boxes_rescale, 1)
        boxes = boxes_rescale.tolist()
        boxes_rescale_yxyx = boxes_rescale.squeeze(1).numpy()
        boxes_rescale_xyxy = boxes_rescale_yxyx[:, [1,0,3,2]]
        boxes_rescale_xywh = xyxy2xywh(boxes_rescale_xyxy)
    else:
        boxes_rescale_xyxy = []
        boxes_rescale_xywh = []

    boxes_GT, label_data_corr_xyxy, label_data_corr_xywh, label_data_corr_yxyx, label_data_class, label_data_class_names\
        = ut.load_gt_labels(img, label_path, class_names_gt, class_names_sel)
    
    # Positive box corr
    for i, box in enumerate(boxes_GT):
        for j in range(4):
            if box[0][j] < 0:
                boxes_GT[i][0][j] = 0

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

    # Empty saliency maps
    if saliency_maps_orig_all[target_indices_pred[0]].sum() == 0:
        return True
    
    """Optimize ODAM Faithfulness"""

    ### Calculate AI Performance
    if len(boxes): # NOTE: For ODAM only consider the GT bbox and prediced bbox for one specific target
        Vacc = ut.calculate_acc(boxes_rescale_xywh[target_indices_pred], label_data_corr_xywh[[target_idx_GT]]) # NOTE: only 1 "GT" target
    else:
        Vacc = 0

    target_prob = []
    target_prob_exp = []
    odam_output_dir = args.output_dir.replace('fullgradcamraw','odam')
    output_path = os.path.join(odam_output_dir,img_name)
    os.makedirs(odam_output_dir, exist_ok=True)
               
    for i, (bbox, cls_name, class_prob) in enumerate(zip(boxes, class_names, class_prob_list)):
        if cls_name[0] in class_names_sel:
            target_prob.append([class_prob])
            if i in target_indices_pred:
                target_prob_exp.append([class_prob])

    """Searching the optimzal sigma"""
    if os.path.exists(os.path.join(args.output_dir.replace('fullgradcamraw','optimization'),f"{img_path.split('/')[-1].split('.')[0]}.json")):
        record = json.load(open(os.path.join(args.output_dir.replace('fullgradcamraw','optimization'),f"{img_path.split('/')[-1].split('.')[0]}.json"),'r'))
        best_sigma = record['best_sigma']
        best_faithfulness = record['best_faithfulness']
        logging.info(f"CUDA{args.device} ({args.object}) [{args.target_layer}] {img_name}: [Read in saved result] best faithfulness={round(best_faithfulness,3)} at sigma={round(best_sigma,3)}")
    else:

        best_sigma = None
        best_faithfulness = 0

        sigma = 1
        exp_box_w, exp_box_h = boxes_rescale_xywh[target_indices_pred][0][2],boxes_rescale_xywh[target_indices_pred][0][3]
        sigma_high = min(max(exp_box_w, exp_box_h), 200)
        # sigma_high = min(max(height, width) / 4, 200)

        record = {
            "attempts" : defaultdict(dict),
            # "bisection" : defaultdict(dict),
        }

        start = time.time()

        faithfulness_epsilon = 0.01
        sigma_precision = 4

        best_step = 0

        step = 0
        while sigma < sigma_high:
            sigma += sigma_precision

            # NOTE: only calculate target object's map when optimizing ODAM faithfulness for the sake of time
            masks_attempt, masks_sum_attempt = rescale_and_apply_gaussian([saliency_maps_orig_all[target_indices_pred[0]]], mapped_locs, height, width, sigma, saliency_method.sel_norm_str)
            
            # Calculate faithfulness for both high and mid sigma
            masks_ndarray = masks_attempt[0].squeeze().detach().cpu().numpy()
            preds_deletion, preds_insertation, _, imgs_deletion, imgs_insertion = compute_faith(model, img, masks_ndarray,label_data_class, cfg)
            dAUC = mean_valid_confidence(label_data_corr_xyxy[[target_idx_GT]], [boxes_rescale_xyxy[target_indices_pred]] + preds_deletion[0], [[prob[0] for prob in target_prob_exp]] + preds_deletion[4]) # include step 0 on intact image
            iAUC = mean_valid_confidence(label_data_corr_xyxy[[target_idx_GT]], preds_insertation[0], preds_insertation[4]) # FIXME: edge case return empty preds_deletion or preds_insertation earlier
            faithfulness = iAUC + (1-dAUC)
            record['attempts'][step]["sigma"] = sigma
            record['attempts'][step]["iAUC"] = iAUC
            record['attempts'][step]["dAUC"] = dAUC
            record['attempts'][step]["faithfulness"] = faithfulness

            if faithfulness > best_faithfulness:
                best_faithfulness = faithfulness
                best_sigma = sigma
                best_step = step

            step += 1
        
        end = time.time()

        logging.info(f"CUDA{args.device} ({args.object}) [{args.target_layer}] {img_name}: best faithfulness={round(best_faithfulness,3)} at sigma={round(best_sigma,3)} in {step} steps ({round(end-start,2)}s)")

        # Record
        record["best_sigma"] = best_sigma
        record["best_faithfulness"] = best_faithfulness
        record["search_seconds"] = end-start
        os.makedirs(args.output_dir.replace('fullgradcamraw','optimization'), exist_ok=True)
        json.dump(record, open(os.path.join(args.output_dir.replace('fullgradcamraw','optimization'),f"{img_path.split('/')[-1].split('.')[0]}.json"),'w'))

    # Saving
    masks, masks_sum = rescale_and_apply_gaussian(saliency_maps_orig_all, mapped_locs, height, width, best_sigma, saliency_method.sel_norm_str)
    
    masks_ndarray = masks[target_indices_pred[0]].squeeze().detach().cpu().numpy()

    # AI Saliency Map Computation
    preds_deletion, preds_insertation, _, imgs_deletion, imgs_insertion = compute_faith(model, img, masks_ndarray,label_data_class, cfg)

    # dAUC = mean_valid_confidence(label_data_corr_xyxy[[target_idx_GT]], [boxes_rescale_xyxy[target_indices_pred]] + preds_deletion[0], [[prob[0] for prob in target_prob]] + preds_deletion[4]) # include step 0 on intact image
    # iAUC = mean_valid_confidence(label_data_corr_xyxy[[target_idx_GT]], preds_insertation[0], preds_insertation[4])
    
    if save_visualization:
        ### Display
        # Generate whole-image saliency maps as reference        
        all_mask_img = result.copy()
        all_mask_img, heat_map = ut.get_res_img(masks_sum, all_mask_img)

        res_img = result.copy()
        for i, mask in enumerate(masks):
            if i in target_indices_pred:
                res_img, heat_map = ut.get_res_img(mask, res_img)

        for i, (bbox, cls_name, class_prob) in enumerate(zip(boxes, class_names, class_prob_list)):
            if cls_name[0] in class_names_sel:
                if i in target_indices_pred:
                    res_img = ut.put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2], res_img) / 255
                    all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2], all_mask_img) / 255
                    
                    res_img = ut.put_text_box(target_bbox_GT, "GT BB for EXP", res_img, color=(255,0,0)) / 255
                else:
                    all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2], all_mask_img, color=(0,0,255)) / 255

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
            images = [cv2.imread(f"{path_config.get('Paths','data_dir')}/mscoco/images/resized/EXP/{img_path.split('/')[-1]}")]
        else:
            images = [cv2.imread(f"{path_config.get('Paths','data_dir')}/bdd/{args.object}_exp/{img_path.split('/')[-1]}")]
        images.append(gt_img * 255)

        # no matching prediction and therefore empty saliency maps
        if len(target_indices_pred) == 0:
            images.append(res_img) # original image
        else:
            images.append(res_img * 255)

        images.append(all_mask_img * 255)
        final_image = ut.concat_images(images)
        cv2.imwrite(output_path, final_image)

        # compress gif
        # downscaled_ratio = 0.4
        # saved_size = [int(width * downscaled_ratio),
        #             int(height * downscaled_ratio)
        #             ]
        # saliency_preview = [cv2.resize((res_img * 255).astype('uint8')[...,::-1], saved_size) for i in range(6)]
        # imgs_deletion_new = saliency_preview + [cv2.resize(img_orig, saved_size) for img_orig in imgs_deletion]
        # imgs_insertion_new = saliency_preview + [cv2.resize(img_orig, saved_size) for img_orig in imgs_insertion]
        # imageio.mimsave(f'{os.path.join(odam_output_dir,img_name+".deletion")}.gif', imgs_deletion_new, duraion=500)
        # imageio.mimsave(f'{os.path.join(odam_output_dir,img_name+".insertion")}.gif', imgs_insertion_new, duraion=500)

    gc.collect()
    torch.cuda.empty_cache()

    # Saving
    mdict={'masks_ndarray': masks_ndarray,
            'sigma': best_sigma,
            'layer': args.target_layer,
            'boxes_pred_xyxy': boxes_rescale_xyxy[target_indices_pred],
            'boxes_pred_xywh': boxes_rescale_xywh[target_indices_pred],
            'boxes_gt_xywh': label_data_corr_xywh[[target_idx_GT]],
            'boxes_gt_xyxy': label_data_corr_xyxy[[target_idx_GT]],
            'HitRate': Vacc,
            'preds_deletion': np.array(preds_deletion,dtype='object'),
            'preds_insertation': np.array(preds_insertation,dtype='object'),
            'boxes_pred_conf': target_prob_exp,
            'boxes_pred_class_names': class_names,
            'class_names_sel': class_names_sel,
            'boxes_gt_classes_names': label_data_class_names,
            'grad_act': raw_data,
            'target_indices_pred': target_indices_pred,
            }
    torch.save(mdict, output_path + '.pth')

    """FullGradCAM"""

    output_path = os.path.join(args.output_dir,img_name)
    os.makedirs(args.output_dir, exist_ok=True)

    ### Calculate AI Performance
    if len(boxes):
        Vacc = ut.calculate_acc(boxes_rescale_xywh, label_data_corr_xywh)/len(boxes_GT)
    else:
        Vacc = 0

    if save_visualization:

        ### Display
        res_img = result.copy()
        res_img, heat_map = ut.get_res_img(masks_sum, res_img)
        for i, (bbox, cls_name, class_prob) in enumerate(zip(boxes, class_names, class_prob_list)):
            if cls_name[0] in class_names_sel:
                #bbox, cls_name = boxes[0][i], class_names[0][i]
                # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
                res_img = ut.put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2], res_img) / 255

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
        cv2.imwrite(output_path, final_image)

    gc.collect()
    torch.cuda.empty_cache()

    masks_ndarray = masks_sum.squeeze().detach().cpu().numpy()

    # AI Saliency Map Computation
    preds_deletion, preds_insertation, _, imgs_deletion, imgs_insertion = compute_faith(model, img, masks_ndarray, label_data_class, cfg)

    # dAUC = mean_valid_confidence(label_data_corr_xyxy, [boxes_rescale_xyxy] + preds_deletion[0], [[prob[0] for prob in target_prob]] + preds_deletion[4]) # include step 0 on intact image
    # iAUC = mean_valid_confidence(label_data_corr_xyxy, preds_insertation[0], preds_insertation[4])

    # compress gif
    # downscaled_ratio = 0.4
    # saved_size = [int(width * downscaled_ratio),
    #               int(height * downscaled_ratio)
    #             ]
    # saliency_preview = [cv2.resize((res_img * 255).astype('uint8')[...,::-1], saved_size) for i in range(6)]
    # imgs_deletion_new = saliency_preview + [cv2.resize(img_orig, saved_size) for img_orig in imgs_deletion]
    # imgs_insertion_new = saliency_preview + [cv2.resize(img_orig, saved_size) for img_orig in imgs_insertion]
    # imageio.mimsave(f'{os.path.join(args.output_dir,img_name+".deletion")}.gif', imgs_deletion_new, duraion=500)
    # imageio.mimsave(f'{os.path.join(args.output_dir,img_name+".insertion")}.gif', imgs_insertion_new, duraion=500)


    # Saving
    mdict={'masks_ndarray': masks_ndarray,
            'sigma': best_sigma,
            'layer': args.target_layer,
            'boxes_pred_xyxy': boxes_rescale_xyxy,
            'boxes_pred_xywh': boxes_rescale_xywh,
            'boxes_gt_xywh': label_data_corr_xywh,
            'boxes_gt_xyxy': label_data_corr_xyxy,
            'HitRate': Vacc,
            'preds_deletion': np.array(preds_deletion,dtype='object'),
            'preds_insertation': np.array(preds_insertation,dtype='object'),
            'boxes_pred_conf': target_prob,
            'boxes_pred_class_names': class_names,
            'class_names_sel': class_names_sel,
            'boxes_gt_classes_names': label_data_class_names,
            'grad_act': raw_data,
            'target_indices_pred': target_indices_pred,
            }

    torch.save(mdict, output_path + '.pth')

    """Activation Maps ODAM"""

    odam_output_dir = args.output_dir.replace('rpn_saliency_maps','rpn_activation_maps').replace('fullgradcamraw','odam')
    output_path = os.path.join(odam_output_dir,img_name)
    os.makedirs(odam_output_dir, exist_ok=True)

    # Saving
    masks, masks_sum = rescale_and_apply_gaussian(activation_maps_orig_all, mapped_locs, height, width, best_sigma, saliency_method.sel_norm_str)
    
    masks_ndarray = masks[target_indices_pred[0]].squeeze().detach().cpu().numpy()
    
    if save_visualization:
        ### Display
        # Generate whole-image saliency maps as reference        
        all_mask_img = result.copy()
        all_mask_img, heat_map = ut.get_res_img(masks_sum, all_mask_img)

        res_img = result.copy()
        for i, mask in enumerate(masks):
            if i in target_indices_pred:
                res_img, heat_map = ut.get_res_img(mask, res_img)

        for i, (bbox, cls_name, class_prob) in enumerate(zip(boxes, class_names, class_prob_list)):
            if cls_name[0] in class_names_sel:
                if i in target_indices_pred:
                    res_img = ut.put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2], res_img) / 255
                    all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2], all_mask_img) / 255
                    
                    res_img = ut.put_text_box(target_bbox_GT, "GT BB for EXP", res_img, color=(255,0,0)) / 255
                else:
                    all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2], all_mask_img, color=(0,0,255)) / 255

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
            images = [cv2.imread(f"{path_config.get('Paths','data_dir')}/mscoco/images/resized/EXP/{img_path.split('/')[-1]}")]
        else:
            images = [cv2.imread(f"{path_config.get('Paths','data_dir')}/bdd/{args.object}_exp/{img_path.split('/')[-1]}")]
        images.append(gt_img * 255)

        # no matching prediction and therefore empty saliency maps
        if len(target_indices_pred) == 0:
            images.append(res_img) # original image
        else:
            images.append(res_img * 255)

        images.append(all_mask_img * 255)
        final_image = ut.concat_images(images)
        cv2.imwrite(output_path, final_image)

    # Saving
    mdict={'masks_ndarray': masks_ndarray,
            'sigma': best_sigma,
            'layer': args.target_layer,
            'boxes_pred_xyxy': boxes_rescale_xyxy[target_indices_pred],
            'boxes_pred_xywh': boxes_rescale_xywh[target_indices_pred],
            'boxes_gt_xywh': label_data_corr_xywh[[target_idx_GT]],
            'boxes_gt_xyxy': label_data_corr_xyxy[[target_idx_GT]],
            'HitRate': Vacc,
            'boxes_pred_conf': target_prob_exp,
            'boxes_pred_class_names': class_names,
            'class_names_sel': class_names_sel,
            'boxes_gt_classes_names': label_data_class_names,
            'grad_act': raw_data,
            'target_indices_pred': target_indices_pred,
            }
    torch.save(mdict, output_path + '.pth')

    """Activation Maps FullGradCAM"""

    output_path = os.path.join(args.output_dir,img_name).replace('rpn_saliency_maps','rpn_activation_maps')
    os.makedirs(args.output_dir.replace('rpn_saliency_maps','rpn_activation_maps'), exist_ok=True)

    ### Calculate AI Performance
    if len(boxes):
        Vacc = ut.calculate_acc(boxes_rescale_xywh, label_data_corr_xywh)/len(boxes_GT)
    else:
        Vacc = 0

    if save_visualization:

        ### Display
        res_img = result.copy()
        res_img, heat_map = ut.get_res_img(masks_sum, res_img)
        for i, (bbox, cls_name, class_prob) in enumerate(zip(boxes, class_names, class_prob_list)):
            if cls_name[0] in class_names_sel:
                #bbox, cls_name = boxes[0][i], class_names[0][i]
                # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
                res_img = ut.put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2], res_img) / 255

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
        cv2.imwrite(output_path, final_image)

    gc.collect()
    torch.cuda.empty_cache()

    masks_ndarray = masks_sum.squeeze().detach().cpu().numpy()

    # Saving
    mdict={'masks_ndarray': masks_ndarray,
            'sigma': best_sigma,
            'layer': args.target_layer,
            'boxes_pred_xyxy': boxes_rescale_xyxy,
            'boxes_pred_xywh': boxes_rescale_xywh,
            'boxes_gt_xywh': label_data_corr_xywh,
            'boxes_gt_xyxy': label_data_corr_xyxy,
            'HitRate': Vacc,
            'boxes_pred_conf': target_prob,
            'boxes_pred_class_names': class_names,
            'class_names_sel': class_names_sel,
            'boxes_gt_classes_names': label_data_class_names,
            'grad_act': raw_data,
            'target_indices_pred': target_indices_pred,
            }

    torch.save(mdict, output_path + '.pth')

    return False

if __name__ == '__main__':

    args = parser.parse_args()

    if args.object == "vehicle":
        args.model_path = f"{path_config.get('Paths','model_dir')}/model_final_721ade.pkl"
        args.method = "fullgradcamraw"
        args.prob = "class"
        args.output_main_dir = f"{path_config.get('Paths','result_dir')}/bdd2coco/rpn_saliency_maps_fasterrcnn/fullgradcamraw_vehicle"
        args.coco_labels = f"{path_config.get('Paths','data_dir')}/mscoco/annotations/COCO_classes2.txt"
        args.img_path = f"{path_config.get('Paths','data_dir')}/bdd/orib_veh_id_task0922"
        args.label_path = f"{path_config.get('Paths','data_dir')}/bdd/orib_veh_id_task0922_mscoco_label"

    elif args.object == "human":
        args.model_path = f"{path_config.get('Paths','model_dir')}/model_final_721ade.pkl"
        args.method = "fullgradcamraw"
        args.prob = "class"
        args.output_main_dir = f"{path_config.get('Paths','result_dir')}/bdd2coco/rpn_saliency_maps_fasterrcnn/fullgradcamraw_human"
        args.coco_labels = f"{path_config.get('Paths','data_dir')}/mscoco/annotations/COCO_classes2.txt"
        args.img_path = f"{path_config.get('Paths','data_dir')}/bdd/orib_hum_id_task1009"
        args.label_path = f"{path_config.get('Paths','data_dir')}/bdd/orib_hum_id_task1009_mscoco_label"

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    device = f"cuda"
    save_visualization = args.visualize

    device = args.device

    failed_imgs = [] # TODO: save failed images into json

    if args.object == 'vehicle':
        input_size = (608, 608)
        sampled_images = ['362.jpg','930.jpg','117.jpg']
        skip_images = [] #['1007.jpg', '1023.jpg', '1028.jpg', '1041.jpg', '1079.jpg', '1108.jpg', '1121.jpg', '1127.jpg', '1170.jpg', '1201.jpg', '1253.jpg', '1258.jpg', '1272.jpg', '134.jpg', '1344.jpg', '1356.jpg', '210.jpg', '297.jpg', '321.jpg', '355.jpg', '383.jpg', '390.jpg', '406.jpg', '425.jpg', '485.jpg', '505.jpg', '52.jpg', '542.jpg', '634.jpg', '648.jpg', '711.jpg', '777.jpg', '784.jpg', '796.jpg', '797.jpg', '838.jpg', '848.jpg', '857.jpg', '899.jpg', '902.jpg', '953.jpg', '967.jpg', '969.jpg', '988.jpg', '99.jpg', '993.jpg']
    elif args.object == 'human':
        input_size = (608, 608)
        sampled_images = ['47.jpg','601.jpg','1304.jpg']
        skip_images = [] #['2334.jpg', '1313.jpg', '1302.jpg', '2186.jpg', '1770.jpg', '1154.jpg', '1663.jpg', '186.jpg', '425.jpg', '875.jpg', '845.jpg', '829.jpg', '388.jpg', '748.jpg', '900.jpg', '1346.jpg', '1803.jpg', '1359.jpg', '1022.jpg', '97.jpg', '2203.jpg', '1066.jpg', '231.jpg', '1097.jpg', '488.jpg', '415.jpg', '2128.jpg', '2008.jpg', '2121.jpg', '2092.jpg', '2271.jpg', '1506.jpg', '1389.jpg', '1954.jpg', '2226.jpg', '670.jpg', '2161.jpg', '1041.jpg', '250.jpg', '1141.jpg', '348.jpg', '1063.jpg', '452.jpg', '601.jpg', '19.jpg', '1746.jpg', '1917.jpg', '1420.jpg', '1817.jpg', '270.jpg', '1398.jpg', '2040.jpg', '11.jpg', '1475.jpg', '897.jpg', '1805.jpg', '997.jpg', '1788.jpg']

    # default vehicle
    class_names_sel = ['car', 'bus', 'truck']
    # args.model_path = 'yolov5sbdd100k300epoch.pt'
    if args.object=='human':
        class_names_sel = ['person']

    class_names_gt = [line.strip() for line in open(args.coco_labels)]

    parser = argparse.ArgumentParser()
    arguments, _ = get_parser(os.path.join(args.img_path, 'test.jpg'), 'cuda', args.model_path).parse_known_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(arguments))
    cfg = setup_cfg(arguments)
    # 构建模型
    model = build_model(cfg)
    # 加载权重
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    img_list = os.listdir(args.img_path)
    img_list.sort()
    label_list = os.listdir(args.label_path)
    label_list.sort()
    # print(img_list)
    for item_img, item_label in zip(img_list[int(args.img_start):int(args.img_end)], label_list[int(args.img_start):int(args.img_end)]):

        if item_img in skip_images or item_img in failed_imgs: continue # model failed to detect the target
        # if item_img not in sampled_images: continue

        for i, (target_layer_group_name,layer_param) in enumerate(target_layer_group_dict.items()):
            # if target_layer_group_name not in ['F15','F16','F17']: continue

            if i < args.layer_start or i >= args.layer_end:
                continue

            if target_layer_group_name == 'stem.MaxPool' or\
                target_layer_group_name == 'backbone.stem.conv1': continue
            
            # if 'backbone' in target_layer_group_name: continue
            # if target_layer_group_name not in ['roi_heads.pooler.level_poolers.0','roi_heads.res5.0.conv1',]: continue
            
            sub_dir_name = args.method + '_' + args.object + '_' + sel_nms + '_' + args.prob + '_' + target_layer_group_name + '_' + sel_faith + '_' + args.norm + '_' + args.model_path.split('/')[-1][:-3].replace('.','') + '_' + '1'
            args.output_dir = os.path.join(args.output_main_dir, sub_dir_name)
            args.target_layer = target_layer_group_name

            if os.path.exists(os.path.join(args.output_dir, f"{split_extension(item_img,suffix='-res')}.pth")):
                continue

            saliencyMap_method = GradCAM(net=model, layer_name=args.target_layer, 
                                        class_names=class_names_gt, sel_classes=class_names_sel, 
                                        sel_norm_str=sel_norm,
                                        sel_method=args.method, layers=target_layer_group_dict)
            try:
                failed = main(os.path.join(args.img_path, item_img), os.path.join(args.label_path, item_label), target_layer_group_name, 
                                model, saliencyMap_method,
                                arguments, cfg, args,
                                class_names_gt, class_names_sel,
                                save_visualization)
                if failed:
                    logging.warning(f"CUDA{args.device} ({args.object}) [{target_layer_group_name}] {item_img}: Skipped due to zero faithfulness")
                    failed_imgs.append(item_img)
            except Exception as e:
                if "out of memory" in str(e):
                    logging.error(f"CUDA{args.device} ({args.object}) [{target_layer_group_name}] {item_img}: GPU out of memory")
                    del saliencyMap_method, model
                    gc.collect()
                    torch.cuda.empty_cache()                         
                    sys.exit(1)
                else:
                    logging.exception(f"CUDA{args.device} ({args.object}) [{target_layer_group_name}] {item_img}: Runtime error")

            del saliencyMap_method
            gc.collect()
            torch.cuda.empty_cache()                         