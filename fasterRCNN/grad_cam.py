# -*- coding: utf-8 -*-
"""
 @File    : grad_cam.py
 @Time    : 2020/3/14 下午4:06
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import utils_previous as ut
import math,gc

def roi_map_to_original_image(pool_dams, rois, original_size):
    """
    Map pooled feature data/saliency maps back to the original image size, ensuring ROI does not exceed image boundaries.
    Reference: https://github.com/Cyang-Zhao/ODAM/blob/e776c3b69050038cbf6a640e5d7b78a930458341/model/rcnn_odamTrain/network.py#L238

    :param pool_dams: Tensor of shape [N, 1, H_pool, W_pool], the pooled feature data.
    :param rois: Tensor of shape [N, 4] (x1, y1, x2, y2) based on original image dimensions.
    :param original_size: Tuple (height, width) of the original image size.
    :return: Tensor of shape [N, original_size[0], original_size[1]], mapped feature data.
    """
    N, c, h_pool, w_pool = pool_dams.shape
    assert c == 1 # have already sum over channels when calculating saliency maps
    images = torch.zeros((N, c, original_size[0], original_size[1]), device=pool_dams.device)

    for i in range(N):
        # Extract ROI and clamp coordinates to ensure they stay within the image boundaries
        x1, y1, x2, y2 = rois[i]
        x1 = x1.clamp(min=0, max=original_size[1] - 1).floor()
        y1 = y1.clamp(min=0, max=original_size[0] - 1).floor()
        x2 = x2.clamp(min=x1 + 1, max=original_size[1]).ceil()
        y2 = y2.clamp(min=y1 + 1, max=original_size[0]).ceil()

        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        target_h, target_w = y2 - y1, x2 - x1

        # Resize the pooled features to match the size of the clamped ROI
        resized_feature = F.interpolate(
            pool_dams[i].unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False)

        # Place the resized features into the corresponding location on the blank original-sized image
        images[i, 0, y1:y2, x1:x2] = resized_feature

    return images

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

class GradCAM:
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, class_names, sel_norm_str, sel_method):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self.class_names = class_names
        self.sel_norm_str = sel_norm_str
        self.sel_XAImethod = sel_method
        self.sel_classes = ['car', 'bus', 'truck']


    def _get_features_hook(self, module, input, output):
        self.feature = output    #self.feature = output
        print("feature shape:{}".format(self.feature.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def generate_saliency_map(self, inputs, target_box, prob_thresh=0.5, grid_size=(16, 16), n_masks=5000, seed=0):
        np.random.seed(seed)
        # image_h, image_w = image.shape[:2]
        image = inputs['image'].unsqueeze(0)
        image_w, image_h = image.size(3), image.size(2)
        res = np.zeros((image_h, image_w), dtype=np.float32)

        for _ in range(n_masks):
            mask = generate_mask(image_size=(image_w, image_h), grid_size=grid_size, prob_thresh=prob_thresh)
            masked = mask_image(image, mask)
            inputs['image'] = masked.squeeze(0)
            with torch.no_grad():
                output = self.net.inference([inputs])
                pred_list = output[0]['instances'].pred_boxes.tensor[:, [1, 0, 3, 2]].tolist()      # yxyx
                pred_class = [self.class_names[i] for i in output[0]['instances'].pred_classes]
                score_list = output[0]['instances'].scores.tolist()

            max_score = 0
            for bbox, cls, score in zip(pred_list, pred_class, score_list):
                if cls in self.sel_classes:
                    iou_score = bbox_iou(target_box, bbox) * score
                    max_score = max(max_score, iou_score)

            res += mask * max_score

            # print(_, max_score)
            # 删除不再使用的变量
            del masked, pred_list, pred_class, score_list
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

    # def __call__(self, inputs, index=0):
    def forward(self, inputs, index=0):

        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        output = self.net.inference([inputs])
        saliency_maps = []
        class_prob_list = []
        head_num_list = []
        raw_data_rec = []
        nObj = 0
        c, h, w = inputs['image'].size()
        pred_list = []
        pred_list.append([])
        pred_list.append([])
        pred_list.append([])
        for output_score, box_corr, class_id in zip(output[0]['instances'].scores, output[0]['instances'].pred_boxes.tensor, output[0]['instances'].pred_classes):
            if self.class_names[class_id] in self.sel_classes:
            # if self.class_names[class_id] == 'person' or self.class_names[class_id] == 'rider':
                # print(output)
                score = output_score                                        #output[0]['instances'].scores[index]
                # proposal_idx = output[0]['instances'].indices[index]  # box来自第几个proposal
                self.net.zero_grad()
                score.backward(retain_graph=True)

                class_prob_score = score
                class_prob_score = torch.unsqueeze(class_prob_score, 0)
                class_prob_list.append(class_prob_score)
                box_corr_t = box_corr[[1,0,3,2]]    #xyxy->yxyx
                # box_corr_t = box_corr
                bbox = box_corr_t.tolist()
                pred_list[0].append([bbox])
                cls = class_id.cpu().data.numpy()
                pred_list[1].append([cls])
                cls_name = self.class_names[class_id]
                pred_list[2].append([cls_name])



                gradients = self.gradient
                activations = self.feature

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
                elif self.sel_XAImethod == 'DRISE':
                    res = self.generate_saliency_map(inputs, bbox, prob_thresh=0.5, grid_size=(16, 16), n_masks=2000,
                                                     seed=0)
                    res_expanded = np.expand_dims(np.expand_dims(res, axis=0), axis=0)
                    saliency_map = torch.tensor(res_expanded, device=inputs['image'].device)
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
                    saliency_map = F.relu((gradients * activations).sum(1, keepdim=True))#.sum(0, keepdim=True))

                # print(saliency_map[0].size())

                # For ROI pooling layer's saliency map, combine all proposals 
                #   and map to correspondinglocation of the image
                if saliency_map.size()[0] != 1: # more than 1 maps due to different proposals
                    # with torch.no_grad():
                    proposals = self.net.inference([inputs],do_postprocess=False,output_proposals=True)[0].proposal_boxes.tensor.detach()
                    saliency_map = roi_map_to_original_image(saliency_map, proposals, (h,w)).sum(0,keepdim=True)
                else: # Backbone layers, directly interpolate to image size
                    saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                if self.sel_norm_str == 'norm':
                    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                    saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

                nObj = nObj + 1
                if nObj == 1:
                    saliency_map_sum = saliency_map
                else:
                    saliency_map_sum = saliency_map_sum + saliency_map

                saliency_maps.append(saliency_map.detach().cpu())


        if nObj == 0:
            saliency_map_sum = torch.zeros([1, 1, h, w])
            saliency_maps.append(torch.zeros([1, 1, h, w]))
        else:
            saliency_map_sum = saliency_map_sum / nObj

        # b, k, u, v = gradients.size()
        # alpha = gradients.view(b, k, -1).mean(2)
        # weights = alpha.view(b, k, 1, 1)
        # saliency_map = (weights * activations).sum(1, keepdim=True)
        # saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        # saliency_map_sum = F.relu(saliency_map_sum)

        saliency_map_sum_min, saliency_map_sum_max = saliency_map_sum.min(), saliency_map_sum.max()
        saliency_map_sum = (saliency_map_sum - saliency_map_sum_min).div(saliency_map_sum_max - saliency_map_sum_min).data
        saliency_map_sum = saliency_map_sum.detach().cpu()

        # if nObj > 0 and output[0]['instances'].scores.numel():
        #     score.backward()
        # else:
        #     print('No Score')

        gc.collect()
        torch.cuda.empty_cache()

        FrameStack = np.empty((len(raw_data_rec),), dtype=np.object)
        for i in range(len(raw_data_rec)):
            FrameStack[i] = raw_data_rec[i]

        if len(head_num_list) > 0:
            class_prob_list = torch.cat(class_prob_list).cpu().detach().numpy()

        return saliency_maps, saliency_map_sum, pred_list, class_prob_list, FrameStack


# class GradCamPlusPlus(GradCAM):
#     def __init__(self, net, layer_name):
#         super(GradCamPlusPlus, self).__init__(net, layer_name)
#
#     def __call__(self, inputs, index=0):
#         """
#
#         :param inputs: {"image": [C,H,W], "height": height, "width": width}
#         :param index: 第几个边框
#         :return:
#         """
#         self.net.zero_grad()
#         output = self.net.inference([inputs])
#         print(output)
#         score = output[0]['instances'].scores[index]
#         proposal_idx = output[0]['instances'].indices[index]  # box来自第几个proposal
#         score.backward()
#
#         gradient = self.gradient[proposal_idx].cpu().data.numpy()  # [C,H,W]
#         gradient = np.maximum(gradient, 0.)  # ReLU
#         indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
#         norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
#         for i in range(len(norm_factor)):
#             norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
#         alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]
#
#         weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)
#
#         feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]
#
#         cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
#         cam = np.sum(cam, axis=0)  # [H,W]
#         # cam = np.maximum(cam, 0)  # ReLU
#
#         # 数值归一化
#         cam -= np.min(cam)
#         cam /= np.max(cam)
#         # resize to box scale
#         box = output[0]['instances'].pred_boxes.tensor[index].detach().numpy().astype(np.int32)
#         x1, y1, x2, y2 = box
#         cam = cv2.resize(cam, (x2 - x1, y2 - y1))
#
#         return cam

    def __call__(self, input_img):

        return self.forward(input_img)