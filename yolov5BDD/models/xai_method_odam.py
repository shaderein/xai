import time
import torch
import torch.nn.functional as F
import gc
import utils.util_my_yolov5 as ut
import numpy as np

import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_yolo_layer(model, layer_name):
    """Find yolov5 layer to calculate GradCAM and GradCAM++

    Args:
        model: yolov5 model.
        layer_name (str): the name of layer with its hierarchical information.

    Return:
        target_layer: found layer
    """
    hierarchy = layer_name.split('_')
    target_layer = model.model._modules[hierarchy[0]]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
    return target_layer


def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask

def mask_image(input_img, mask):
    # masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
    #           255).astype(np.uint8)
    device = input_img.device
    mask = torch.tensor(mask, dtype=torch.float32, device=device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand_as(input_img)
    output_img = input_img * mask

    # output_img_np = output_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
    # plt.imshow(output_img_np)
    # plt.axis('off')
    # plt.show()

    return output_img


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


class YOLOV5XAI:

    def __init__(self, model, layer_names, sel_prob_str, sel_norm_str, sel_classes, sel_XAImethod, img_size=(640, 640)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        self.sel_prob_str = sel_prob_str
        self.sel_norm_str = sel_norm_str
        self.sel_classes = sel_classes
        self.sel_XAImethod = sel_XAImethod

        def backward_hook_0(module, grad_input, grad_output):
            self.gradients[0] = grad_output[0]
            return None
        def forward_hook_0(module, input, output):
            self.activations[0] = output
            return None

        def backward_hook_1(module, grad_input, grad_output):
            self.gradients[1] = grad_output[0]
            return None
        def forward_hook_1(module, input, output):
            self.activations[1] = output
            return None

        def backward_hook_2(module, grad_input, grad_output):
            self.gradients[2] = grad_output[0]
            return None
        def forward_hook_2(module, input, output):
            self.activations[2] = output
            return None

        target_layer = find_yolo_layer(self.model, layer_names[0])
        target_layer.register_forward_hook(forward_hook_0)
        target_layer.register_backward_hook(backward_hook_0)
        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations[0].shape[2:])

        target_layer = find_yolo_layer(self.model, layer_names[1])
        target_layer.register_forward_hook(forward_hook_1)
        target_layer.register_backward_hook(backward_hook_1)
        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations[1].shape[2:])

        target_layer = find_yolo_layer(self.model, layer_names[2])
        target_layer.register_forward_hook(forward_hook_2)
        target_layer.register_backward_hook(backward_hook_2)
        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations[2].shape[2:])

    def generate_saliency_map(self, image, target_box, prob_thresh=0.5, grid_size=(16, 16), n_masks=5000, seed=0):
        np.random.seed(seed)
        # image_h, image_w = image.shape[:2]
        image_w, image_h = image.size(3), image.size(2)
        res = np.zeros((image_h, image_w), dtype=np.float32)

        for _ in range(n_masks):


            mask = generate_mask(image_size=(image_w, image_h), grid_size=grid_size, prob_thresh=prob_thresh)
            masked = mask_image(image, mask)

            with torch.no_grad():
                preds, logits, preds_logits, classHead_output = self.model(masked)
            pred_list = preds[0][0]
            pred_class = preds[2][0]
            score_list = preds[3][0]

            max_score = 0
            for bbox, cls, score in zip(pred_list, pred_class, score_list):
                if cls in self.sel_classes:
                    iou_score = bbox_iou(target_box, bbox) * score
                    max_score = max(max_score, iou_score)

            res += mask * max_score

            print(_)
            # 删除不再使用的变量
            del masked, preds, logits, preds_logits, classHead_output
            # 释放显存
            torch.cuda.empty_cache()

            # # 将处理后的图片移动到 cpu 上并转换为 numpy 数组，调整维度顺序为 (height, width, channels)
            # input_img_np = input_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
            # # 创建一个新的matplotlib图表
            # fig, ax = plt.subplots(1)
            # # 显示图像
            # ax.imshow(input_img_np)
            # # 获取边界框的坐标
            # y1, x1, y2, x2 = bbox
            # # 创建一个 Rectangle patch
            # rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            # # 将矩形添加到图表中
            # ax.add_patch(rect)
            # # 显示图像和边界框
            # plt.axis('off')
            # plt.show()


        return res

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        class_prob_list = []
        head_num_list = []
        nObj = 0
        b, c, h, w = input_img.size()
        tic = time.time()
        preds, logits, preds_logits, classHead_output = self.model(input_img)
        classHead_output = classHead_output[0].detach().numpy()
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        # class_probs = torch.sigmoid(logits[0])

        raw_data_rec = []
        class_probs = (logits[0])
        pred_list = []
        pred_list.append([])
        pred_list.append([])
        pred_list.append([])
        pred_list.append([])
        with torch.autograd.set_detect_anomaly(True):
            for pred_logit, logit, bbox, cls, cls_name, obj_prob, class_prob, classHead in zip(preds_logits[0], logits[0], preds[0][0], preds[1][0], preds[2][0], preds[3][0], class_probs, classHead_output):
                if cls_name in self.sel_classes:
                    if class_idx:
                        if self.sel_prob_str == 'obj':
                            score = pred_logit
                        else:
                            score = logit[cls]
                        class_prob_score = class_prob[cls]
                        class_prob_score = torch.unsqueeze(class_prob_score, 0)
                        class_prob_list.append(class_prob_score)
                        pred_list[0].append([bbox])     # bbox: yxyx
                        pred_list[1].append([cls])
                        pred_list[2].append([cls_name])
                        pred_list[3].append([obj_prob])
                    else:
                        score = logit.max()
                    self.model.zero_grad()
                    tic = time.time()
                    score.backward(retain_graph=True)
                    print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')

                    head_num_list.append([classHead])
                    gradients = self.gradients[classHead]
                    activations = self.activations[classHead]

                    # img_w, img_h = input_img.size(3), input_img.size(2)
                    # mask = generate_mask(image_size=(img_w, img_h),
                    #                      grid_size=(16, 16),
                    #                      prob_thresh=0.5)
                    # masked_img = masked_img(input_img, mask)

                    # # 将处理后的图片移动到 cpu 上并转换为 numpy 数组，调整维度顺序为 (height, width, channels)
                    # input_img_np = input_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
                    # # 创建一个新的matplotlib图表
                    # fig, ax = plt.subplots(1)
                    # # 显示图像
                    # ax.imshow(input_img_np)
                    # # 获取边界框的坐标
                    # y1, x1, y2, x2 = bbox
                    # # 创建一个 Rectangle patch
                    # rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                    # # 将矩形添加到图表中
                    # ax.add_patch(rect)
                    # # 显示图像和边界框
                    # plt.axis('off')
                    # plt.show()


                    if self.sel_XAImethod == 'gradcam':
                        saliency_map = ut.gradcam_operation(activations, gradients)
                    elif self.sel_XAImethod == 'gradcampp':
                        saliency_map = ut.gradcampp_operation(activations, gradients)
                    elif self.sel_XAImethod == 'fullgradcam':
                        saliency_map = ut.fullgradcam_operation(activations, gradients)
                    elif self.sel_XAImethod == 'fullgradcamraw':
                        saliency_map = ut.fullgradcamraw_operation(activations, gradients)
                    elif self.sel_XAImethod == 'saveRawGradAct':
                        saliency_map, raw_data = ut.saveRawGradAct_operation(activations, gradients)
                        raw_data_rec.append(raw_data)
                    elif self.sel_XAImethod == 'saveRawAllAct':
                        saliency_map = ut.gradcampp_operation(activations, gradients)
                    elif self.sel_XAImethod == 'DRISE':
                        res = self.generate_saliency_map(input_img, bbox, prob_thresh=0.5, grid_size=(16, 16), n_masks=5000, seed=0)
                        res_expanded = np.expand_dims(np.expand_dims(res, axis=0), axis=0)
                        saliency_map = torch.tensor(res_expanded, device=input_img.device)
                        # # 将 tensor 图片转为 numpy 数组，并调整通道位置
                        # input_img_np = input_img.squeeze().permute(1, 2, 0).cpu().numpy()
                        # # 将显著图 res 归一化
                        # res_normalized = (res - res.min()) / (res.max() - res.min())
                        # # 创建图表
                        # plt.figure(figsize=(12, 6))
                        # # 显示原图
                        # plt.imshow(input_img_np)
                        # # 以半透明的方式添加显著图
                        # plt.imshow(res_normalized, cmap='jet', alpha=0.5)
                        # # 显示合并后的图像
                        # plt.axis('off')
                        # plt.show()
                    elif self.sel_XAImethod == 'odam':
                        saliency_map = F.relu((gradients * activations).sum(1, keepdim=True))

                        # # Instance-based FullGradCAM
                        # weights = F.relu(gradients)
                        # saliency_map = (weights * activations).sum(1, keepdim=True)
                        # saliency_map = F.relu(saliency_map)


                    saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                    if self.sel_norm_str == 'norm':
                        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

                    nObj = nObj + 1
                    if nObj == 1:
                        saliency_map_sum = saliency_map
                    else:
                        saliency_map_sum = saliency_map_sum + saliency_map

                    # if self.sel_XAImethod == 'odam':
                    saliency_maps.append(saliency_map.detach().cpu())

                # pred_logit = pred_logit.detach()
                # logit = logit.detach()
                # class_prob = class_prob.detach()

        if nObj == 0:
            saliency_map_sum = torch.zeros([1, 1, h, w])
            saliency_maps.append(torch.zeros([1, 1, h, w]))
        else:
            saliency_map_sum = saliency_map_sum / nObj

        # if self.sel_XAImethod != 'odam':
        #     #saliency_map_sum = F.relu(saliency_map_sum)
        #     saliency_map = saliency_map_sum
        #     # saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        #     # saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        #     saliency_map = saliency_map.detach().cpu()
        #     saliency_maps.append(saliency_map)

        saliency_map_sum = saliency_map_sum.detach().cpu()

        if preds_logits[0].numel():
            score = pred_logit
            score.backward()
        # if logits[0].numel():
        #     score = logit[0]
        #     score.backward()

        self.activations[0] = self.activations[0].detach().cpu().numpy()
        self.activations[1] = self.activations[1].detach().cpu().numpy()
        self.activations[2] = self.activations[2].detach().cpu().numpy()
        if self.sel_XAImethod == 'saveRawAllAct':
            raw_data_rec = self.activations.copy()
        else:
            raw_data_rec = []
        if nObj != 0:
            # self.activations[0] = self.activations[0].detach().cpu()
            # self.activations[1] = self.activations[1].detach().cpu()
            # self.activations[2] = self.activations[2].detach().cpu()
            pred_logit = pred_logit.detach().cpu()
            logit = logit.detach().cpu()
            class_prob = class_prob.detach().cpu()
            activations = activations.detach().cpu()
            gradients = gradients.detach().cpu()

        gc.collect()
        torch.cuda.empty_cache()

        FrameStack = np.empty((len(raw_data_rec),), dtype=np.object)
        for i in range(len(raw_data_rec)):
            FrameStack[i] = raw_data_rec[i]

        if len(head_num_list) > 0:
            head_num_list = np.squeeze(head_num_list,1).astype(int)
            class_prob_list = torch.cat(class_prob_list).cpu().detach().numpy()

        return torch.cat(saliency_maps), saliency_map_sum, pred_list, class_prob_list, head_num_list, FrameStack

    def __call__(self, input_img):

        return self.forward(input_img)
