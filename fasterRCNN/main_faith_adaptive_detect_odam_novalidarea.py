import warnings
warnings.filterwarnings("ignore")

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os,re

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
from grad_cam_head import GradCAM, bbox_iou        #, GradCamPlusPlus
from skimage import io
from torch import nn
from utils_previous import get_res_img, put_text_box, concat_images, calculate_acc, scale_coords_new, xyxy2xywh, xywh2xyxy
import util_my_yolov5 as ut

import imageio

import argparse
from deep_utils import Box, split_extension
import scipy.io
import pandas as pd
# from numba import cuda
from GPUtil import showUtilization as gpu_usage
import gc
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math, time

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
    'roi_heads.pooler.level_poolers.0' : [],
    'roi_heads.res5.0.conv1' : [1,2,0],
    'roi_heads.res5.0.conv2' : [3,1,1],
    'roi_heads.res5.0.conv3' : [1,1,0],
    'roi_heads.res5.1.conv1' : [1,1,0],
    'roi_heads.res5.1.conv2' : [3,1,1],
    'roi_heads.res5.1.conv3' : [1,1,0],
    'roi_heads.res5.2.conv1' : [1,1,0],
    'roi_heads.res5.2.conv2' : [3,1,1],
    'roi_heads.res5.2.conv3' : [1,1,0],
    }


# Arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--model-path', type=str, default=sel_model, help='Path to the model')
# parser.add_argument('--img-path', type=str, default=input_main_dir, help='input image path')
parser.add_argument('--output-dir', type=str, default='sample_EM_idtask_1_output_update_2/GradCAM_NMS_objclass_F0_singleScale_norm_v5s_1', help='output dir')
# parser.add_argument('--output-main-dir', type=str, default=output_main_dir, help='output root dir')
parser.add_argument('--img-size', type=int, default=608, help="input image size")
parser.add_argument('--target-layer', type=list, default='F1',
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')

parser.add_argument('--method', type=str, default="odam", help='gradcam or eigencam or eigengradcam or weightedgradcam or gradcampp or fullgradcam')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
# parser.add_argument('--label-path', type=str, default=input_main_dir_label, help='input label path')

parser.add_argument('--object', type=str, default="vehicle", help='human or vehicle')
parser.add_argument('--prob', type=str, default="class", help='obj, class, objclass')
parser.add_argument('--coco-labels', type=str, default="COCO_classes.txt", help='path to coco classes list')

args = parser.parse_args()


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


def get_parser(img_path, run_device, dataset):
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="/mnt/h/jinhan/xai/fasterRCNN/faster_rcnn_R_50_C4_1x.yaml", #"configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
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
    if dataset == 'mscoco':
        model_path = "/mnt/h/jinhan/xai/models/model_final_721ade.pkl"
    else:
        model_path =  "/mnt/h/jinhan/xai/models/FasterRCNN_C4_BDD100K.pth"
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", model_path, "MODEL.DEVICE", run_device],
        nargs=argparse.REMAINDER,
    )
    return parser

def compute_faith(model, img, masks_ndarray, label_data_class, label_data_corr_xywh, cfg):
    # Compute Region Area
    # valid_area = 0
    # for cor in label_data_corr_xywh:
    #     valid_area = valid_area + cor[2]*cor[3]
    valid_area = np.sum(masks_ndarray>1e-2)
    valid_area = np.round(valid_area).astype('int32')

    height, width = img.shape[:2]
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image = transform_gen.get_transform(img).apply_image(img)
    torch_img = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)

    # Jinhan: DEBUG
    imgs_deletion = []
    imgs_insertation = []

    # torch_img = model.preprocessing(img[..., ::-1])

    ### Deletion
    # Sort Saliency Map
    delta_thr = 0
    num_thr = 100
    masks_ndarray[np.isnan(masks_ndarray)] = 0
    masks_ndarray[masks_ndarray <= delta_thr] = 0
    if sum(sum(masks_ndarray)) == 0:
        masks_ndarray[0, 0] = 1
        masks_ndarray[1, 1] = 0.5
        return [],[],[]
    masks_ndarray_flatten = masks_ndarray.flatten()
    # masks_ndarray_positive = masks_ndarray_flatten[masks_ndarray_flatten > delta_thr]
    masks_ndarray_positive = masks_ndarray_flatten
    masks_ndarray_sort = masks_ndarray_positive
    masks_ndarray_sort.sort()   # ascend
    masks_ndarray_sort = masks_ndarray_sort[::-1]   # descend
    masks_ndarray_sort = masks_ndarray_sort[:valid_area]
    masks_ndarray_RGB = np.expand_dims(masks_ndarray, 2)
    masks_ndarray_RGB = np.concatenate((masks_ndarray_RGB, masks_ndarray_RGB, masks_ndarray_RGB),2)
    # thr_ascend = np.linspace(masks_ndarray_sort[0], masks_ndarray_sort[-1], num_thr, False)
    thr_idx = np.floor(np.linspace(0, masks_ndarray_sort.size, num_thr, False))
    thr_descend = masks_ndarray_sort[thr_idx.astype('int')]
    thr_ascend = thr_descend[::-1]
    img_raw = img
    img_raw_float = img_raw.astype('float')/255
    device = 'cuda' #if next(model.model.parameters()).is_cuda else 'cpu'
    preds_deletion_rec = [] #torch.zeros(0, torch_img.size(1), torch_img.size(2), torch_img.size(3), device=device)
    input_batch = []
    with torch.no_grad():
        for i_thr in thr_descend:
            img_raw_float_use = img_raw_float.copy()
            img_raw_float_use[masks_ndarray_RGB > i_thr] = np.random.rand(sum(sum(sum(masks_ndarray_RGB > i_thr))), )
            img_raw_uint8_use = (img_raw_float_use*255).astype('uint8')

            imgs_deletion.append(img_raw_uint8_use[..., ::-1]) # Jinhan: save image to view deletion process

            # torch_img_rand = model.preprocessing(img_raw_uint8_use[..., ::-1])
            image = transform_gen.get_transform(img_raw_uint8_use).apply_image(img_raw_uint8_use)
            torch_img_rand = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": torch_img_rand, "height": height, "width": width}
            input_batch.append(inputs)
            if not (input_batch.__len__() % 10):
                preds_deletion_rec.extend(model(input_batch))
                input_batch = []
        if input_batch.__len__():
            preds_deletion_rec.extend(model(input_batch))
            input_batch = []
    preds_deletion = []
    for preds_deletion_rec_i in preds_deletion_rec:
        preds_deletion.append(preds_deletion_rec_i['instances'])

    shape_raw = [torch_img_rand.size(2), torch_img_rand.size(1)]  # w, h
    shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
    # Jinhan: unlike Yolov5, FasterRCNN doesn't have head indices in its output, but keep the return value ds consistent
    #   for the downstream faithfulness mAUC calculation
    pred_deletion_adj = [[[] for _ in range(num_thr)] for _ in range(5)]
    for i, (preds_deletion_i) in enumerate(preds_deletion):
        for bbox_one, cls_idx_one, conf_one in zip(preds_deletion_i.pred_boxes.tensor, preds_deletion_i.pred_classes, preds_deletion_i.scores):
            if cls_idx_one.item() in label_data_class.astype(np.int64):
                
                # Jinhan: DEBUG
                # imgs_deletion[i] = put_text_box(bbox_one,'test',imgs_deletion[i]).astype('uint8')[...,[1,0,2]]
                
                boxes_rescale_xyxy, boxes_rescale_xywh, _ = rescale_box_list([[bbox_one.detach().cpu().numpy()[[1,0,3,2]]]], shape_new, shape_new) # yxyx
                pred_deletion_adj[0][i].append(boxes_rescale_xyxy.tolist()[0])
                pred_deletion_adj[1][i].append(boxes_rescale_xywh.tolist()[0])
                pred_deletion_adj[2][i].append(cls_idx_one.detach().cpu().numpy())
                pred_deletion_adj[4][i].append(conf_one.detach().cpu().numpy())

    ## Show Examples
    # show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 99]
    # img_show_torch = torch_img_deletion_rec[show_idx,:,:,:]
    # img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    # img_show_ndarray_cat1 = np.concatenate((img_show_ndarray[:,:,:,0], img_show_ndarray[:,:,:,1], img_show_ndarray[:,:,:,2]), axis=1)
    # img_show_ndarray_cat2 = np.concatenate((img_show_ndarray[:,:,:,3], img_show_ndarray[:,:,:,4], img_show_ndarray[:,:,:,5]), axis=1)
    # img_show_ndarray_cat3 = np.concatenate((img_show_ndarray[:,:,:,6], img_show_ndarray[:,:,:,7], img_show_ndarray[:,:,:,8]), axis=1)
    # img_show_ndarray_cat = np.concatenate((img_show_ndarray_cat1, img_show_ndarray_cat2, img_show_ndarray_cat3), axis=0)
    # plt.figure()
    # plt.imshow(img_show_ndarray_cat)
    # plt.show()

    ### Insertation
    preds_insertation_rec = [] #torch.zeros(0, torch_img.size(1), torch_img.size(2), torch_img.size(3), device=device)
    input_batch = []
    with torch.no_grad():
        for i_thr in thr_descend:
            img_raw_float_use = img_raw_float.copy()
            img_raw_float_use[masks_ndarray_RGB <= i_thr] = 0
            img_raw_uint8_use = (img_raw_float_use*255).astype('uint8')

            # Jinhan : DEBUG
            imgs_insertation.append(img_raw_uint8_use[..., ::-1]) # Jinhan: save image to view deletion process

            # torch_img_rand = model.preprocessing(img_raw_uint8_use[..., ::-1])
            image = transform_gen.get_transform(img_raw_uint8_use).apply_image(img_raw_uint8_use)
            torch_img_rand = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": torch_img_rand, "height": height, "width": width}
            input_batch.append(inputs)
            if not (input_batch.__len__() % 10):
                preds_insertation_rec.extend(model(input_batch))
                input_batch = []
        if input_batch.__len__():
            preds_insertation_rec.extend(model(input_batch))
            input_batch = []
    preds_insertation = []
    for preds_insertation_rec_i in preds_insertation_rec:
        preds_insertation.append(preds_insertation_rec_i['instances'])

    pred_insertation_adj = [[[] for _ in range(num_thr)] for _ in range(5)]
    for i, (preds_insertation_i) in enumerate(preds_insertation):
        for bbox_one, cls_idx_one, conf_one in zip(preds_insertation_i.pred_boxes.tensor, preds_insertation_i.pred_classes, preds_insertation_i.scores):
            if cls_idx_one.item() in label_data_class.astype(np.int64):
                
                # Jinhan: DEBUG
                # imgs_insertation[i] = put_text_box(bbox_one,'test',imgs_insertation[i]).astype('uint8')[...,[1,0,2]]
                
                boxes_rescale_xyxy, boxes_rescale_xywh, _ = rescale_box_list([[bbox_one.detach().cpu().numpy()[[1,0,3,2]]]], shape_new, shape_new) # yxyx
                pred_insertation_adj[0][i].append(boxes_rescale_xyxy.tolist()[0])
                pred_insertation_adj[1][i].append(boxes_rescale_xywh.tolist()[0])
                pred_insertation_adj[2][i].append(cls_idx_one.detach().cpu().numpy())
                pred_insertation_adj[4][i].append(conf_one.detach().cpu().numpy())


    ## Show Examples
    # show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 99]
    # img_show_torch = torch_img_insertation_rec[show_idx,:,:,:]
    # img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    # img_show_ndarray_cat1 = np.concatenate((img_show_ndarray[:,:,:,0], img_show_ndarray[:,:,:,1], img_show_ndarray[:,:,:,2]), axis=1)
    # img_show_ndarray_cat2 = np.concatenate((img_show_ndarray[:,:,:,3], img_show_ndarray[:,:,:,4], img_show_ndarray[:,:,:,5]), axis=1)
    # img_show_ndarray_cat3 = np.concatenate((img_show_ndarray[:,:,:,6], img_show_ndarray[:,:,:,7], img_show_ndarray[:,:,:,8]), axis=1)
    # img_show_ndarray_cat = np.concatenate((img_show_ndarray_cat1, img_show_ndarray_cat2, img_show_ndarray_cat3), axis=0)
    # plt.figure()
    # plt.imshow(img_show_ndarray_cat)
    # plt.show()

    return pred_deletion_adj, pred_insertation_adj, thr_descend, imgs_deletion, imgs_insertation

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

def main(arguments, img_path, label_path, target_layer_group, model, cfg, img_num,
        dataset, class_names_gt, class_names_sel,
        sel_norm="norm", sel_method="odam"):
    # sel_norm_str = 'norm'


    # setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.info("Arguments: " + str(arguments))
    #
    cfg = setup_cfg(arguments)
    # print(cfg)
    # # 构建模型
    # model = build_model(cfg)
    # # 加载权重
    # checkpointer = DetectionCheckpointer(model)
    # checkpointer.load(cfg.MODEL.WEIGHTS)

    # 加载图像
    path = os.path.expanduser(arguments.input)
    img = read_image(path, format="BGR")
    height, width = img.shape[:2]
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image = transform_gen.get_transform(img).apply_image(img)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)

    inputs = {"image": image, "height": height, "width": width}

    # 获取类别名称
    meta = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )

    if dataset=='vehicle':
        bb_selections = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/bdd/labels_mapping/Random_sample_vehicle_procedure_analysis.xlsx','veh_sample_img_condition')
        bb_selections = bb_selections[['image','vehicle_count_gt','ExpTargetIndex']]
        bb_selection = bb_selections.loc[bb_selections['image']==img_path.split('/')[-1]] # 1029.jpg
    elif dataset=='human':
        bb_selections = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/bdd/labels_mapping/Random_sample_human_procedure_analysis.xlsx','hum_sample_img_condition')
        bb_selections = bb_selections[['imgnumber','human_count_gt','ExpTargetIndex']]
        bb_selection = bb_selections.loc[bb_selections['imgnumber']==int(img_path.split('/')[-1].replace('.jpg',''))] # 1029.jpg
    elif dataset == 'mscoco':
        # Find instance used in experiments
        bb_selections = pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/mscoco/other/for_eyegaze_GT_infos_size_ratio.xlsx')
        bb_selection = bb_selections.loc[bb_selections['img']==img_path.split('/')[-1].replace('.jpg','')] # horse_382088.png

        # class used in experiments
        class_name = re.sub(r"_\d+\.(jpg|png)",'',item_img).replace('_',' ')
        if class_name not in class_names_gt:
            print(f'[WARNING] {item_img} category parsed as {class_name}')
            return
        class_names_sel = [class_name] # generate saliency maps for specific category

    # Grad-CAM
    # layer_name = get_last_conv_name(model)
    layer_name = target_layer_group
    saliencyMap_method = GradCAM(net=model, layer_name=layer_name, class_names=class_names_gt, sel_norm_str=sel_norm,
                                 sel_method=sel_method, layers=target_layer_group_dict, sigma_factors=sigma_factors)
    saliencyMap_method.sel_classes = class_names_sel
    masks_all, masks_sum_all, [boxes, _, class_names], class_prob_list, raw_data = saliencyMap_method(inputs)  # cam mask
    
    saliencyMap_method.remove_handlers()

    for sigma_factor in masks_all:

        masks = masks_all[sigma_factor]
        masks_sum = masks_sum_all[sigma_factor]

        ### Convert Image
        torch_img = image.unsqueeze(0)
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr

        ### New Images
        result_raw = result
        images = [img]
        result = img
        
        if sigma_factor != -1:
            for i in range(len(masks)):
                # DEBUG
                masks[i] = masks[i] #F.interpolate(masks[i], size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
            masks_sum = masks_sum #F.interpolate(masks_sum, size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
        else:
            for i in range(len(masks)):
                # DEBUG
                masks[i] = F.interpolate(masks[i], size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
                saliency_map_min, saliency_map_max = masks[i].min(), masks[i].max()
                masks[i] = (masks[i] - saliency_map_min).div(saliency_map_max - saliency_map_min).data

            masks_sum = F.interpolate(masks_sum, size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = masks_sum.min(), masks_sum.max()
            masks_sum = (masks_sum - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        if sel_method != 'odam':    
            masks = [masks_sum]

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
            boxes_rescale_xyxy = 0
            boxes_rescale_xywh = 0

        # boxes[0][0] = [322, 0.0, 381, 35]
        # boxes[1][0] = [322, 0.0, 381, 35]
        # boxes[0][0] = [576, 0, 100, 200]

        ### Load labels
        # label_data = np.loadtxt(label_path, dtype=np.float32, delimiter=' ')
        # if len(label_data.shape) == 1:
        #     label_data = label_data[None, :]
        # label_data_class = label_data[:, 0]
        # label_data_corr = label_data[:, 1:]
        # label_data_corr = label_data_corr[label_data_class < 2, :]  # filter classes
        # label_data_class = label_data_class[label_data_class < 2]  # filter class labels
        # img_h, img_w = np.size(img, 0), np.size(img, 1)
        # label_data_corr[:, 0] = label_data_corr[:, 0] * img_w
        # label_data_corr[:, 1] = label_data_corr[:, 1] * img_h
        # label_data_corr[:, 2] = label_data_corr[:, 2] * img_w
        # label_data_corr[:, 3] = label_data_corr[:, 3] * img_h
        # label_data_corr_xywh = label_data_corr
        # label_data_corr = xywh2xyxy(label_data_corr)
        # label_data_corr_xyxy = label_data_corr
        # label_data_corr_yxyx = label_data_corr[:, [1, 0, 3, 2]]
        # label_data_corr_yxyx = np.round(label_data_corr_yxyx)
        # boxes_GT = label_data_corr_yxyx[:, None, :].tolist()

        boxes_GT, label_data_corr_xyxy, label_data_corr_xywh, label_data_corr_yxyx, label_data_class, label_data_class_names\
            = ut.load_gt_labels(img, label_path, class_names_gt, class_names_sel)
        
        # Positive box corr
        for i, box in enumerate(boxes_GT):
            for j in range(4):
                if box[0][j] < 0:
                    boxes_GT[i][0][j] = 0

        # Generalize to both whole-image and instance-based saliency maps
        if sel_method == 'odam':
            # Find best matching target BB's index from pred (y1,x1,y2,x2)
            # For MSCOCO: Chenyang provided both GT target index and coordinates x1,y1,x2,y2
            if dataset == 'mscoco':
                target_bbox_GT = [bb_selection[['y1']].y1.item()*img.shape[0],
                                bb_selection[['x1']].x1.item()*img.shape[1],
                                bb_selection[['y2']].y2.item()*img.shape[0],
                                bb_selection[['x2']].x2.item()*img.shape[1]]
                boxes_GT_overlaps = [area(box_GT[0],target_bbox_GT,threshold=0.1) for box_GT in boxes_GT]
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
        else:
            # preserve all prediction boxes
            target_indices_pred = range(len(boxes)) #FIXME

        ### Calculate AI Performance
        if len(boxes): # NOTE: For ODAM only consider the GT bbox and prediced bbox for one specific target
            Vacc = calculate_acc(boxes_rescale_xywh[target_indices_pred], label_data_corr_xywh[[target_idx_GT]]) # NOTE: only 1 "GT" target
        else:
            Vacc = 0

        ### Display

        # Generate whole-image saliency maps as reference        
        all_mask_img = result.copy()
        all_mask_img, heat_map = ut.get_res_img(masks_sum, all_mask_img)

        res_img = result.copy()
        for i, mask in enumerate(masks):
            if i in target_indices_pred:
                res_img, heat_map = get_res_img(mask, res_img)
        obj_prob = []
        for i, (bbox, cls_name, class_prob) in enumerate(zip(boxes, class_names, class_prob_list)):
            if cls_name[0] in class_names_sel:
                if i in target_indices_pred:
                    # bbox, cls_name = boxes[0][i], class_names[0][i]
                    # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
                    res_img = put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2],
                                        res_img) / 255
                    all_mask_img = put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2],
                        all_mask_img) / 255
                    
                    # FIXME: check EXP BB selection
                    res_img = ut.put_text_box(target_bbox_GT, "GT BB for EXP", res_img, color=(255,0,0)) / 255


                    obj_prob.append([class_prob.cpu().detach().numpy()[0]])
                else:
                    all_mask_img = ut.put_text_box(bbox[0], cls_name[0] + ", " + str(class_prob.cpu().detach().numpy()[0] * 100)[:2],
                        all_mask_img, color=(0,0,255)) / 255

        ## Display Ground Truth
        gt_img = result.copy()
        gt_img = gt_img / gt_img.max()
        for i, (bbox, cls_idx) in enumerate(zip(boxes_GT, label_data_class)):
            cls_idx = np.int8(cls_idx)
            if class_names_gt[cls_idx] in class_names_sel:
                if i==target_idx_GT:
                    # bbox, cls_name = boxes[0][i], class_names[0][i]
                    # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
                    gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img, color=(0,255,0)) / 255
                else:
                    gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img, color=(0,0,255)) / 255

        # images.append(gt_img * 255)
        if dataset == 'mscoco':
            images = [cv2.imread(f"/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/EXP/{img_path.split('/')[-1]}")]
            images.append(gt_img * 255)
        else:
            images = [gt_img * 255]

        # no matching prediction and therefore empty saliency maps
        if len(target_indices_pred) == 0:
            images.append(res_img) # original image
        else:
            images.append(res_img * 255)

        images.append(all_mask_img * 255)
        final_image = concat_images(images)
        img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
        if sigma_factor == -1:
            output_path = f'{args.output_dir.replace("gaussian_sigmaFACTOR","bilinear")}/{img_name}'
            os.makedirs(args.output_dir.replace("gaussian_sigmaFACTOR", "bilinear"), exist_ok=True)
        else:
            output_path = f'{args.output_dir.replace("FACTOR",str(sigma_factor))}/{img_name}'
            os.makedirs(args.output_dir.replace("FACTOR",str(sigma_factor)), exist_ok=True)
        print(f'[INFO] Saving the final image at {output_path}')
        cv2.imwrite(output_path, final_image)

        gc.collect()
        torch.cuda.empty_cache()

        # # AI Saliency Map Computation
        if len(target_indices_pred) == 0:
            masks_ndarray = np.zeros(masks[0].squeeze().detach().cpu().numpy().shape)
        else: 
            masks_ndarray = masks[target_indices_pred[0]].squeeze().detach().cpu().numpy()

        # preds_deletion, preds_insertation, _ = compute_faith(model, img, masks_ndarray,label_data_class, label_data_corr_xywh[[target_idx_GT]], cfg)
        
        preds_deletion, preds_insertation, _, imgs_deletion, imgs_insertion = compute_faith(model, img, masks_ndarray,label_data_class, label_data_corr_xywh[[target_idx_GT]], cfg)
        # compress gif
        downscaled_ratio = 0.4
        saved_size = [int(res_img.shape[1] * downscaled_ratio),
                      int(res_img.shape[0] * downscaled_ratio)
                    ]
        saliency_preview = [cv2.resize((res_img * 255).astype('uint8')[...,::-1], saved_size) for i in range(60)]
        imgs_deletion_new = saliency_preview + [cv2.resize(img_orig, saved_size) for img_orig in imgs_deletion]
        imgs_insertion_new = saliency_preview + [cv2.resize(img_orig, saved_size) for img_orig in imgs_insertion]
        if sigma_factor == -1:
            imageio.mimsave(f'{os.path.join(args.output_dir.replace("gaussian_sigmaFACTOR","bilinear"),img_name+".deletion")}.gif', imgs_deletion_new)
            imageio.mimsave(f'{os.path.join(args.output_dir.replace("gaussian_sigmaFACTOR","bilinear"),img_name+".insertion")}.gif', imgs_insertion_new)
        else:
            imageio.mimsave(f'{os.path.join(args.output_dir.replace("FACTOR",str(sigma_factor)),img_name+".deletion")}.gif', imgs_deletion_new)
            imageio.mimsave(f'{os.path.join(args.output_dir.replace("FACTOR",str(sigma_factor)),img_name+".insertion")}.gif', imgs_insertion_new)


        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                    'boxes_pred_xyxy': boxes_rescale_xyxy[target_indices_pred],
                                                    'boxes_pred_xywh': boxes_rescale_xywh[target_indices_pred],
                                                    'boxes_gt_xywh': label_data_corr_xywh[[target_idx_GT]],
                                                    'boxes_gt_xyxy': label_data_corr_xyxy[[target_idx_GT]],
                                                    'HitRate': Vacc,
                                                    'boxes_pred_conf': obj_prob,
                                                    'boxes_pred_class_names': [class_names[idx] for idx in target_indices_pred],
                                                    'preds_deletion': preds_deletion,
                                                    'preds_insertation': preds_insertation,
                                                    'class_names_sel': class_names_sel,
                                                    'boxes_gt_classes_names': [label_data_class_names[target_idx_GT]],
                                                    })

        # # # Human Saliency Map Loading
        # human_saliency_map_path = 'E:/HKU/HKU_XAI_Project/XAI_Similarity_1/human_saliency_map_hum_new_1/' + img_num + '_GSmo_30.mat'
        # human_saliency_map = scipy.io.loadmat(human_saliency_map_path)['output_map_norm']
        # human_deletion, human_insertation, _ = compute_faith(model, img, human_saliency_map, label_data_corr_xywh, cfg)
        #
        # scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
        #                                               'boxes_pred_xyxy': boxes_rescale_xyxy,
        #                                               'boxes_pred_xywh': boxes_rescale_xywh,
        #                                               'boxes_gt_xywh': label_data_corr_xywh,
        #                                               'boxes_gt_xyxy': label_data_corr_xyxy,
        #                                               'HitRate': Vacc,
        #                                               'boxes_pred_conf': obj_prob,
        #                                               'boxes_pred_class_names': class_names,
        #                                               'grad_act': raw_data,
        #                                               'human_deletion': human_deletion,
        #                                               'human_insertation': human_insertation
        #                                               })

        # # # Saliency Map Saving
        # scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
        #                                               'boxes_pred_xyxy': boxes_rescale_xyxy,
        #                                               'boxes_pred_xywh': boxes_rescale_xywh,
        #                                               'boxes_gt_xywh': label_data_corr_xywh,
        #                                               'boxes_gt_xyxy': label_data_corr_xyxy,
        #                                               'HitRate': Vacc,
        #                                               'boxes_pred_conf': obj_prob,
        #                                               'boxes_pred_class_names': class_names,
        #                                               'grad_act': raw_data
        #                                               })

        # TrainedXAI Saliency Map Loading
        # trainedXAI_saliency_map_path = 'E:\HKU\HKU_XAI_Project\Human_Inspired_XAI_Try2_FasterRCNN_detectHuman\saveRawSaliencyMapData_testSet_FasterRCNN_GaussianConv_OnlyAF/' + img_num + '_trainedSaliencyMap.mat'
        # trainedXAI_saliency_map = scipy.io.loadmat(trainedXAI_saliency_map_path)['PredData_raw']
        # trainedXAI_deletion, trainedXAI_insertation, _ = compute_faith(model, img, trainedXAI_saliency_map,
        #                                                                   label_data_corr_xywh, cfg)

        # scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
        #                                               'boxes_pred_xyxy': boxes_rescale_xyxy,
        #                                               'boxes_pred_xywh': boxes_rescale_xywh,
        #                                               'boxes_gt_xywh': label_data_corr_xywh,
        #                                               'boxes_gt_xyxy': label_data_corr_xyxy,
        #                                               'HitRate': Vacc,
        #                                               'boxes_pred_conf': obj_prob,
        #                                               'boxes_pred_class_names': class_names,
        #                                               'grad_act': raw_data,
        #                                               'trainedXAI_deletion': trainedXAI_deletion,
        #                                               'trainedXAI_insertation': trainedXAI_insertation
        #                                               })


# if __name__ == "__main__":
#     """
#     Usage:export KMP_DUPLICATE_LIB_OK=TRUE
#     python detection/demo.py --config-file detection/faster_rcnn_R_50_C4.yaml \
#       --input ./examples/pic1.jpg \
#       --opts MODEL.WEIGHTS /Users/yizuotian/pretrained_model/model_final_b1acc2.pkl MODEL.DEVICE cpu
#     """
#     mp.set_start_method("spawn", force=True)
#     arguments = get_parser().parse_args()
#     main(arguments)

if __name__ == '__main__':
    device = args.device

    print('[INFO] Loading the model')

    sel_method = 'odam'
    sel_nms = 'NMS'
    sel_prob = 'class'
    sel_norm = 'norm'
    sigma_factors = [-1,2,4]

    for category in ["mscoco"]:#,"vehicle","human"]:
        if category == "mscoco":
                class_names_sel = None
                sel_model = '/mnt/h/jinhan/xai/models/model_final_721ade.pkl'
                coco_labels_path = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/COCO_classes.txt"
                class_names_gt = [line.strip() for line in open(coco_labels_path)]
                input_main_dir = '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/DET'   #Veh_id_img
                input_main_dir_label = '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/annotations_DET'   #Veh_id_label
                output_main_dir = '/mnt/h/jinhan/results/mscoco/unrestricted_valid_area/xai_saliency_maps_faster_gaussian_sigmaFACTOR/odam'    # _humanAttention _trainedXAI
        else:
            sel_model = "/mnt/h/jinhan/xai/models/FasterRCNN_C4_BDD100K.pth"
            class_names_gt = ['person', 'rider', 'car', 'bus', 'truck']
            if category == "vehicle":
                class_names_sel = ['car', 'bus', 'truck']
                input_main_dir = "/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_veh_id_task0922"
                input_main_dir_label = "/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_veh_id_task0922_label"
                output_main_dir = "/mnt/h/jinhan/results/bdd/xai_saliency_maps_faster_gaussian_sigmaFACTOR/odam_vehicle"
            elif category == "human":
                class_names_sel = ['person', 'rider']
                input_main_dir = "/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_hum_id_task1009"
                input_main_dir_label = "/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_hum_id_task1009_label"
                output_main_dir = "/mnt/h/jinhan/results/bdd/xai_saliency_maps_faster_gaussian_sigmaFACTOR/odam_human"

        sel_model_str = sel_model.split('/')[-1][:-3].replace('.','')

        arguments = get_parser(os.path.join(input_main_dir, 'test.jpg'), device, category).parse_args()
        setup_logger(name="fvcore")
        logger = setup_logger()
        logger.info("Arguments: " + str(arguments))
        cfg = setup_cfg(arguments)
        # print(cfg)
        # 构建模型
        model = build_model(cfg)
        # 加载权重
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        for i, (target_layer_group_name,layer_param) in enumerate(target_layer_group_dict.items()):
            # if target_layer_group_name not in ['F15','F16','F17']: continue

            if target_layer_group_name == 'stem.MaxPool' or\
                target_layer_group_name == 'backbone.stem.conv1': continue
            
            sub_dir_name = sel_method + '_' + sel_nms + '_' + sel_prob + '_' + target_layer_group_name + '_' + 'singleScale' + '_' + sel_norm + '_' + sel_model_str
            args.output_dir = os.path.join(output_main_dir, sub_dir_name)
            args.target_layer = target_layer_group_name

            if os.path.isdir(input_main_dir):
                img_list = os.listdir(input_main_dir)
                label_list = os.listdir(input_main_dir_label)
                # print(img_list)
                # img_list.reverse()
                # label_list.reverse()
                for item_img in img_list:
                    # if 'chair_81061' not in item_img: continue

                    if category == 'mscoco':
                        sampled_images = ['chair_81061.png','elephant_97230.png','giraffe_287545.png']
                        skip_images = ['book_472678.png','hair drier_178028.png','hair drier_239041.png', 'kite_405279.png', 'mouse_513688.png', 'toaster_232348.png', 'toaster_453302.png', 'toothbrush_218439.png', 'traffic light_453841.png']
                    elif category == 'vehicle':
                        sampled_images = ['362.jpg','930.jpg','1331.jpg']
                    elif category == 'human':
                        sampled_images = ['47.jpg','601.jpg','1304.jpg']

                    if item_img not in sampled_images: continue

                    sigma_factors_to_run = sigma_factors.copy()

                    for sigma_factor in sigma_factors:
                        if sigma_factor == -1:
                            if os.path.exists(os.path.join(args.output_dir.replace('gaussian_sigmaFACTOR','bilinear'), split_extension(item_img,suffix='-res'))) and\
                                os.path.exists(os.path.join(args.output_dir.replace('gaussian_sigmaFACTOR','bilinear'), f"{split_extension(item_img,suffix='-res')}.mat")):
                                sigma_factors_to_run.remove(sigma_factor)
                        elif os.path.exists(os.path.join(args.output_dir.replace('FACTOR',str(sigma_factor)), split_extension(item_img,suffix='-res'))) and\
                            os.path.exists(os.path.join(args.output_dir.replace('FACTOR',str(sigma_factor)), f"{split_extension(item_img,suffix='-res')}.mat")):
                            sigma_factors_to_run.remove(sigma_factor)

                    if len(sigma_factors_to_run)==0: continue
                    if item_img in skip_images: continue # model failed to detect the target

                    item_label = item_img[:-4]+'.txt'
                    arguments = get_parser(os.path.join(input_main_dir, item_img), device, category).parse_args()

                    main(arguments, os.path.join(input_main_dir, item_img), os.path.join(input_main_dir_label, item_label), target_layer_group_name, model, cfg, item_img[:-4],
                        category, class_names_gt, class_names_sel)

                    # del model, saliency_method
                    gc.collect()
                    torch.cuda.empty_cache()                         
                    # gpu_usage()

            else:
                main(input_main_dir)

        del model
