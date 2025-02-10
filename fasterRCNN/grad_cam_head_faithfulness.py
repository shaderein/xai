# -*- coding: utf-8 -*-
"""
 @File    : grad_cam.py
 @Time    : 2020/3/14 下午4:06
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np
import torch, math
import torch.nn.functional as F
import utils_previous as ut
import math,gc
from collections import defaultdict

def roi_map_to_original_image(pool_dams, rois, original_size):
    """
    Map pooled feature data/saliency maps back to the original image size, ensuring ROI does not exceed image boundaries.
    Reference: https://github.com/Cyang-Zhao/ODAM/blob/e776c3b69050038cbf6a640e5d7b78a930458341/model/rcnn_odamTrain/network.py#L238

    :param pool_dams: Tensor of shape [N, 1, H_pool, W_pool], the pooled feature data.
    :param rois: Tensor of shape [N, 4] (x1, y1, x2, y2) based on preprocessed image dimensions.
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

def calculate_receptive_field(layer_name, layers, proposals=None, spatial_scale=0.0625, grid_size=(14,14)):
    """
    Calculate the receptive field, stride, and padding for a specific layer in a CNN.
    Args:
    - layer_name (str): Index of the layer.
    - layers (list): List of layer properties (kernel size, stride, padding).
    - proposals: Tensor of shape [N, 4] (x1, y1, x2, y2) based on original image dimensions
    - spatial scale: parameter of the pooler. Scaling proposal to the feature map coordinate
    - grid_size: (14x14)

    Returns:
    - receptive_field (N, 2): Size of the receptive field in both dimensions
    - jump (N, 2): Jump of the receptive field (in both dimensions). (Product of strides used)
    - start (N, 2): center coordinates of the first feature (top-left)
        (N=1 for the backbone layers)
    """

    proposals_num = 1 if proposals is None else proposals.shape[0]

    receptive_field = np.full((proposals_num, 2), 1.0)
    jump = np.full((proposals_num, 2), 1.0)
    start = np.full((proposals_num, 2), 0.5)

    # TODO: handle the behavior of ROIAlign
    if proposals is not None:
        proposals = proposals.cpu().numpy()
        proposals = proposals * spatial_scale

    for layer in layers:
        if layer == 'roi_heads.pooler.level_poolers.0':
            x1 = proposals[:, 0]
            y1 = proposals[:, 1]
            x2 = proposals[:, 2]
            y2 = proposals[:, 3]
            
            roi_width = x2 - x1
            roi_height = y2 - y1

            locs_in_cell_x = roi_width / grid_size[0]
            locs_in_cell_y = roi_height / grid_size[1]

            receptive_field[:, 0] += (locs_in_cell_x - 1) * jump[:, 0]
            receptive_field[:, 1] += (locs_in_cell_y - 1) * jump[:, 1]

            start[:, 0] = (start[:, 0] + jump[:, 0] * x1) + ((locs_in_cell_x - 1) / 2 * jump[:, 0])
            start[:, 1] = (start[:, 1] + jump[:, 1] * y1) + ((locs_in_cell_y - 1) / 2 * jump[:, 1])

            jump[:, 0] *= locs_in_cell_x
            jump[:, 1] *= locs_in_cell_y

        else:
            kernel_size, layer_stride, layer_padding = layers[layer]

            receptive_field[:, 0] += (kernel_size - 1) * jump[:, 0]
            receptive_field[:, 1] += (kernel_size - 1) * jump[:, 1]

            start[:, 0] += ((kernel_size - 1) / 2 - layer_padding) * jump[:, 0]
            start[:, 1] += ((kernel_size - 1) / 2 - layer_padding) * jump[:, 1]

            jump[:, 0] *= layer_stride
            jump[:, 1] *= layer_stride

        if layer == layer_name:
            break

    return receptive_field, jump, start

def map_saliency_to_original_image(saliency_map, original_size, preprocessed_size, 
                                   jump, start):
    """
    Map saliency map locations to the centers of their receptive fields on the original image.
    Args:
    - saliency_map (ndarray): The saliency map.
    - original_size (tuple): The size of the original image.
    - preprocessed_size (tuple): The size of the preprocessed image.
    - receptive_field (N, 2): Size of the receptive field in both dimensions
    - jump (N, 2): Jump of the receptive field (in both dimensions). (Product of strides used)
    - start (N, 2): center coordinates of the first feature (top-left)

    Returns:
    - mapped_locs (ndarray): Mapped locations on the original image.
    """
    saliency_height, saliency_width = saliency_map.shape[2],saliency_map.shape[3]
    original_height, original_width = original_size
    preprocessed_height, preprocessed_width = preprocessed_size

    height_ratio = original_height / preprocessed_height
    width_ratio = original_width / preprocessed_width

    # Create a grid of indices
    x_indices, y_indices = np.meshgrid(np.arange(saliency_width), np.arange(saliency_height))

    # Expand the indices to match the number of proposals for broadcasting
    x_indices = np.expand_dims(x_indices, axis=0)  # Shape: (1, saliency_height, saliency_width)
    y_indices = np.expand_dims(y_indices, axis=0)

    jump_x = jump[:, 0, np.newaxis, np.newaxis]  # Shape: (num_instances, 1, 1)
    jump_y = jump[:, 1, np.newaxis, np.newaxis]
    start_x = start[:, 0, np.newaxis, np.newaxis]
    start_y = start[:, 1, np.newaxis, np.newaxis]

    # Calculate the center positions in the preprocessed image
    center_x = x_indices * jump_x + start_x # Shape: (1, saliency_height, saliency_width)
    center_y = y_indices * jump_y + start_y

    # Map to original image coordinates
    mapped_center_x = (center_x * width_ratio).astype(int)
    mapped_center_y = (center_y * height_ratio).astype(int)

    # Stack the mapped coordinates into the output array
    mapped_locs = np.stack((mapped_center_x, mapped_center_y), axis=-1)

    return torch.tensor(mapped_locs)

def apply_gaussian_kernel(saliency_map, mapped_locs, sigma, original_size):
    """
    Apply Gaussian kernel to the saliency map locations and upsample to the original image size.
    Args:
    - saliency_map (N,C,H,W): N=#proposals for head and N=1 for backone. C=1 (summed over channls in fullgradcam)
    - mapped_locs (N,H,W,2): Mapped locations on the original image.
    - receptive_field (N,2): Size of the receptive field in x and y dimension
    - original_size (tuple): The size of the original image.

    Returns:
    - output (ndarray): Saliency map upsampled to the original image size.
    """
    N, C, saliency_height, saliency_width = saliency_map.shape
    original_height, original_width = original_size
    output = torch.zeros((N, C, original_height, original_width), dtype=torch.float32, device=saliency_map.device)

    # Generate Gaussian kernels on GPU
    kernel_size_x = int(math.ceil(sigma * 6)) | 1  # Ensure odd size
    kernel_size_y = int(math.ceil(sigma * 6)) | 1

    # Create the 2D Gaussian kernel using meshgrid and broadcasting
    x = torch.arange(kernel_size_x, dtype=torch.float32, device=saliency_map.device) - kernel_size_x // 2
    y = torch.arange(kernel_size_y, dtype=torch.float32, device=saliency_map.device) - kernel_size_y // 2
    gaussian_x = torch.exp(-0.5 * (x ** 2) / sigma ** 2)
    gaussian_y = torch.exp(-0.5 * (y ** 2) / sigma ** 2)
    gaussian_kernel = (gaussian_y[:, None] * gaussian_x[None, :]).unsqueeze(0).unsqueeze(0)

    # Normalize the kernel
    gaussian_kernel /= gaussian_kernel.sum()

    # Flatten saliency map and corresponding locations
    flat_saliency_map = saliency_map.view(N, C, -1)
    center_x = mapped_locs[:, :, :, 0].view(N, -1).long()
    center_y = mapped_locs[:, :, :, 1].view(N, -1).long()


    for n in range(N):
        valid_mask = (center_x[n] >= 0) & (center_x[n] < original_width) & \
                     (center_y[n] >= 0) & (center_y[n] < original_height)

        cx = center_x[n][valid_mask]
        cy = center_y[n][valid_mask]
        saliency_values = flat_saliency_map[n, 0, valid_mask]

        # Compute region bounds for the Gaussian kernel
        sx = torch.clamp(cx - kernel_size_x // 2, 0, original_width)
        ex = torch.clamp(cx + kernel_size_x // 2 + 1, 0, original_width)
        sy = torch.clamp(cy - kernel_size_y // 2, 0, original_height)
        ey = torch.clamp(cy + kernel_size_y // 2 + 1, 0, original_height)

        # Determine relative bounds in the Gaussian kernel
        gsx = torch.clamp(kernel_size_x // 2 - (cx - sx), 0, kernel_size_x)
        gex = gsx + (ex - sx)
        gsy = torch.clamp(kernel_size_y // 2 - (cy - sy), 0, kernel_size_y)
        gey = gsy + (ey - sy)

        # Apply Gaussian kernel across all valid points at once
        for i in range(len(saliency_values)):
            output[n, 0, sy[i]:ey[i], sx[i]:ex[i]] += saliency_values[i] * gaussian_kernel[
                0, 0, gsy[i]:gey[i], gsx[i]:gex[i]
            ]

    return output
class GradCAM:
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, class_names, sel_classes, sel_norm_str, sel_method, layers):
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
        self.sel_classes = sel_classes
        self.layers = layers # layer info

    def _get_features_hook(self, module, input, output):
        self.feature = output    #self.feature = output
        # print("feature shape:{}".format(self.feature.size()))

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

        saliency_maps_orig_all = []
        activation_maps_orig_all = []

        mapped_locs_all, adjusted_receptive_field_all, proposals_all = [], [], []

        saliency_maps = defaultdict(list)
        saliency_map_sum = defaultdict()
        class_prob_list = []
        head_num_list = []
        raw_data_rec = []
        nObj = 0
        c, h, w = inputs['image'].size()
        h_orig = inputs['height']
        w_orig = inputs['width']
        pred_list = []
        pred_list.append([])
        pred_list.append([])
        pred_list.append([])

        output_score, gradients, activations = None, None, None # edge case: no predictions in bdd2coco condition

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

                activation_map = activations.detach().sum(1, keepdim=True)

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

                elif self.sel_XAImethod == 'odam':
                    saliency_map = F.relu((gradients * activations).sum(1, keepdim=True))#.sum(0, keepdim=True))

                saliency_map_orig = saliency_map.detach().cpu()
                activation_map_orig = activation_map.detach().cpu()

                saliency_maps_orig_all.append(saliency_map_orig)
                activation_maps_orig_all.append(activation_map_orig)

                proposals = None

                if saliency_map.size()[0] != 1: # more than 1 maps due to different proposals
                    # Note: proposals have the same coordinates as the preprocessed input image (with fixed height 800)
                    proposals = self.net.inference([inputs],do_postprocess=False,output_proposals=True)[0].proposal_boxes.tensor.detach()
                proposals_all.append(proposals)

                ## Rescale based on receptive field
                receptive_field, jump, start = calculate_receptive_field(self.layer_name, self.layers, proposals)

                # Adjust the receptive field to account for the input resizing in preprocessing
                height_ratio = h_orig / h
                width_ratio = w_orig / w
                adjusted_receptive_field = receptive_field
                adjusted_receptive_field[:,0] = receptive_field[:,0] * width_ratio
                adjusted_receptive_field[:,1] = receptive_field[:,1] * height_ratio

                # print(f"Receptive field: {receptive_field}, adjusted: {adjusted_receptive_field}")

                # Get a grid of coordinates of the receptive field center of each spatial location of the intermediate saliency map, 
                #   mapped to the original image
                mapped_locs = map_saliency_to_original_image(saliency_map_orig, (h_orig,w_orig), (h,w), jump, start)

                mapped_locs_all.append(mapped_locs)
                adjusted_receptive_field_all.append(adjusted_receptive_field)

        self.feature = self.feature.detach().cpu()
        if output_score is not None:
            output_score = output_score.detach().cpu()
        if gradients is not None:
            gradients = gradients.detach().cpu()
        if activations is not None:
            activations = activations.detach().cpu()

        del output_score, gradients, activations

        gc.collect()
        torch.cuda.empty_cache()

        FrameStack = np.empty((len(raw_data_rec),), dtype=np.object)
        for i in range(len(raw_data_rec)):
            FrameStack[i] = raw_data_rec[i]

        if len(head_num_list) > 0:
            class_prob_list = torch.cat(class_prob_list).cpu().detach().numpy()

        return (saliency_maps_orig_all, activation_maps_orig_all),\
                mapped_locs_all,adjusted_receptive_field_all,proposals_all,\
                pred_list, class_prob_list, FrameStack

    def __call__(self, input_img):

        return self.forward(input_img)