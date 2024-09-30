import time, tqdm, re
import torch
import torch.nn.functional as F
import gc
import utils.util_my_yolov5 as ut
import numpy as np
from collections import defaultdict
from functools import partial

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

def calculate_receptive_field(layer_name, layers, upsample_factor=1):
    """
    Calculate the receptive field, stride, and padding for a specific layer in a CNN.
    Args:
    - layer_name (str): Index of the layer.
    - layers (list): List of layer properties (kernel size, stride, padding).
        For yolov5s, e.g. model_2_cv1_act and model_2_cv2_act are parallel and grouped in a dict

    Returns:
    - receptive_field (N, 2): Size of the receptive field in both dimensions
    - jump (N, 2): Jump of the receptive field (in both dimensions). (Product of strides used)
    - start (N, 2): center coordinates of the first feature (top-left)
        (N=1 for the backbone layers)
    """

    receptive_field = np.full((1, 2), 1.0)
    jump = np.full((1, 2), 1.0)
    start = np.full((1, 2), 0.5)

    for layer in layers:

        if isinstance(layers[layer],dict):
            kernel_size, layer_stride, layer_padding = list(layers[layer].values())[0][0]
        else:
            kernel_size, layer_stride, layer_padding = layers[layer][0]

        receptive_field[:, 0] += (kernel_size - 1) * jump[:, 0]
        receptive_field[:, 1] += (kernel_size - 1) * jump[:, 1]

        start[:, 0] += ((kernel_size - 1) / 2 - layer_padding) * jump[:, 0]
        start[:, 1] += ((kernel_size - 1) / 2 - layer_padding) * jump[:, 1]

        jump[:, 0] *= layer_stride
        jump[:, 1] *= layer_stride

        # Adjust jump and start based on upsampling
        if layer in ["model_13","model_17"]:
            upsample_factor = 2
            jump[:, 0] /= upsample_factor
            jump[:, 1] /= upsample_factor
            start[:, 0] /= upsample_factor
            start[:, 1] /= upsample_factor

        # print(f"{layer} r={receptive_field}")

        if isinstance(layers[layer],dict):
            if layer_name in layers[layer]:
                break
        elif layer == layer_name:
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

def apply_gaussian_kernel_with_erf(saliency_map, mapped_locs, receptive_field, original_size, 
                                 effective_mask_prev=None,sigma_factors=[2,4]):
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

    output = dict()
    for sigma_factor in sigma_factors:
        output[sigma_factor] = torch.zeros((N, C, original_height, original_width), dtype=torch.float32, device=saliency_map.device) #TODO

    mapped_saliency_map = torch.zeros((N, C, original_height, original_width), dtype=torch.float32, device=saliency_map.device) # DEBUG

    effective_receptive_field = torch.zeros_like(mapped_locs,dtype=torch.int32)

    # Create a mask the size of the original image
    theoretical_mask = torch.zeros((N, C, original_height, original_width),dtype=torch.int32) # activated theoretical receptive field
    effective_mask = torch.zeros((N, C, original_height, original_width),dtype=torch.int32)

    for n in range(N):
        rx, ry = receptive_field[n]

        # Plot the mapped locations as points on the grid
        for i in range(saliency_height):
            for j in range(saliency_width):
                cx, cy = mapped_locs[n, i, j, 0].item(), mapped_locs[n, i, j, 1].item()

                if cx < 0 or cx >= original_width or cy < 0 or cy >= original_height: continue # skip OOB locations
                if saliency_map[n,0,i,j] <= 0: continue # skip empty locations

                # DEBUG
                mapped_saliency_map[n,0,cy,cx] += saliency_map[n,0,i,j]

                # Calculate the bounds of the theoretical RF
                x_trf_start = max(int(cx - rx // 2), 0)
                x_trf_end = min(int(cx + rx // 2), original_width)
                y_trf_start = max(int(cy - ry // 2), 0)
                y_trf_end = min(int(cy + ry // 2), original_height)

                ## Theoretical RF
                theoretical_mask[n, 0, y_trf_start:y_trf_end, x_trf_start:x_trf_end] = 1

                if effective_mask_prev is None:
                    erx, ery = rx, ry

                else:
                    # skip if no ERF_prev within the current TRF
                    if effective_mask_prev[n, 0, y_trf_start:y_trf_end, x_trf_start:x_trf_end].max() == 0: continue

                    # Filter activated ERF_prev within the current local TRF
                    theoretical_mask_local = torch.zeros((original_height,original_width), dtype=torch.int32)
                    theoretical_mask_local[y_trf_start:y_trf_end, x_trf_start:x_trf_end] = 1
                    effective_mask_prev_local = theoretical_mask_local & effective_mask_prev[n,0]

                    effective_mask_prev_local_pos = torch.nonzero(effective_mask_prev_local, as_tuple=False)

                    x_inds = effective_mask_prev_local_pos[:,1]
                    y_inds = effective_mask_prev_local_pos[:,0]

                    # shifted center when calculating the distance as (1) the RF size is even and (2) the feature center 
                    # (x.5,x.5) is mapped to (int,int) on the original image. These are due to the even kernel size used at L0
                    # TODO: should modify it for fasterRCNN! Or generalize to all models 
                    shifted_cx = cx - 0.5
                    shifted_cy = cy - 0.5

                    # Within the theoretical RF, find max distance to the activated ERF_prev
                    erx = (torch.ceil(torch.max(torch.abs(x_inds - shifted_cx))) * 2).int().item()       
                    ery = (torch.ceil(torch.max(torch.abs(y_inds - shifted_cy))) * 2).int().item()

                    # TODO: generalize to other models?
                    x_erf_start = max(int(cx - erx/2), 0)
                    x_erf_end = min(int(cx + erx/2), original_width)
                    y_erf_start = max(int(cy - ery/2), 0)
                    y_erf_end = min(int(cy + ery/2), original_height)

                    ## ERF
                    effective_mask[n,0,y_erf_start:y_erf_end, x_erf_start:x_erf_end] = 1

                effective_receptive_field[n,i,j,0] = erx
                effective_receptive_field[n,i,j,1] = ery

                ## Apply gaussian (Effective)
                for sigma_factor in sigma_factors:
                    sigma_x = erx / sigma_factor
                    sigma_y = ery / sigma_factor

                    kernel_size_x = math.ceil(sigma_x * 6) # DEBUG: 5.0 * 6 round up to 31?
                    kernel_size_y = math.ceil(sigma_y * 6)
                    if kernel_size_x % 2 == 0: kernel_size_x += 1
                    if kernel_size_y % 2 == 0: kernel_size_y += 1

                    gaussian_x = cv2.getGaussianKernel(ksize=kernel_size_x, sigma=sigma_x)
                    gaussian_y = cv2.getGaussianKernel(ksize=kernel_size_y, sigma=sigma_y)
                    gaussian = gaussian_y * gaussian_x.T  # Create 2D Gaussian kernel
                    h, w = gaussian.shape

                    gaussian = torch.tensor(gaussian, device=saliency_map.device)

                    # Determine the region in the output array
                    sx = np.maximum(cx - w // 2, 0)
                    ex = np.minimum(cx + w // 2 + 1, original_width)
                    sy = np.maximum(cy - h // 2, 0)
                    ey = np.minimum(cy + h // 2 + 1, original_height)

                    # Determine the relative region in the gaussian kernel
                    gsx = np.maximum(0, w // 2 - (cx - sx))
                    gex = gsx + (ex - sx)
                    gsy = np.maximum(0, h // 2 - (cy - sy))
                    gey = gsy + (ey - sy)

                    output[sigma_factor][n, 0, sy:ey, sx:ex] += (
                            saliency_map[n, 0, i, j] * gaussian[gsy:gey, gsx:gex]
                        )
    
    # smoothed maps, ERF mask, TRF mask, ERF size
    if effective_mask_prev is None:
        return output, theoretical_mask, theoretical_mask, effective_receptive_field, mapped_saliency_map
    else:
        return output, effective_mask, theoretical_mask, effective_receptive_field, mapped_saliency_map


class YOLOV5XAI:

    def __init__(self, model, excluded_layers, sel_prob_str, sel_norm_str, sel_classes, sel_XAImethod,layers, sigma_factors=[-1,2,4],device=0,img_size=(608, 608),):
        self.model = model
        self.gradients = defaultdict(defaultdict)
        self.activations = defaultdict(defaultdict)
        self.sel_prob_str = sel_prob_str
        self.sel_norm_str = sel_norm_str
        self.sel_classes = sel_classes
        self.sel_XAImethod = sel_XAImethod
        self.layer_names = []
        self.layers = layers # layer info
        self.sigma_factors = sigma_factors
        self.sel_device = device

        excluded_layers = ["model_9_m_act1","model_9_m_act2","model_9_m_act3"]
        for name, param in layers.items():
            if name in excluded_layers: continue
            if isinstance(param,dict):
                for n,p in param.items():
                    self.layer_names.append(n)
            else:
                self.layer_names.append(name)

        for layer in self.layer_names:
            for head_num in range(3):
                target_layer = find_yolo_layer(self.model, layer)
                # Use partial to pass additional parameters to the hook
                target_layer.register_forward_hook(partial(self.forward_hook, layer=layer, head_num=head_num))
                target_layer.register_backward_hook(partial(self.backward_hook, layer=layer, head_num=head_num))
                device = f'cuda:{self.sel_device}' if next(self.model.model.parameters()).is_cuda else 'cpu'
                self.model(torch.zeros(1, 3, *img_size, device=device))
                # print(f'[{layer}][{head_num}] saliency_map size :', self.activations[layer][head_num].shape[2:])

    def backward_hook(self, module, grad_input, grad_output, layer, head_num):
        self.gradients[layer][head_num] = grad_output[0]
        return None
    def forward_hook(self, module, input, output, layer, head_num):
        self.activations[layer][head_num] = output
        return None   
     
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

            # print(_)
            # 删除不再使用的变量
            del masked, preds, logits, preds_logits, classHead_output
            # 释放显存
            torch.cuda.empty_cache()

        return res

    def forward(self, input_img, shape_orig, sigma_factors=[-1,2,4], class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps= {}
        saliency_map_sum= {}
        activation_maps= {}
        activation_map_sum= {}

        saliency_maps_orig_all = defaultdict(list)
        activation_maps_orig_all = defaultdict(list)

        saliency_maps_orig_mapped_all = defaultdict(list)
        activation_maps_orig_mapped_all = defaultdict(list)

        effective_saliency_masks_all = defaultdict(list) # layer, object
        effective_activation_masks_all = defaultdict(list)
        theoretical_saliency_masks_all = defaultdict(list)
        theoretical_activation_masks_all = defaultdict(list)

        effective_saliency_rf_all = defaultdict(list) # layer, object
        effective_activation_rf_all = defaultdict(list)
        theoretical_rf_all = defaultdict()

        for layer in self.layer_names:
            saliency_maps[layer] = defaultdict(list) # layer, sigma, object, object-based saliency map
            saliency_map_sum[layer] = defaultdict() # layer, sigma, whole-image saliency map

            activation_maps[layer] = defaultdict(list)
            activation_map_sum[layer] = defaultdict()

        class_prob_list = []
        head_num_list = []
        nObj = defaultdict(int) # layer (since some layer has empty saliency map. should normalize separately)
        b, c, h, w = input_img.size()
        h_orig, w_orig = shape_orig
        tic = time.time()
        preds, logits, preds_logits, classHead_output = self.model(input_img)
        classHead_output = classHead_output[0].detach().numpy()
        # print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
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
                    # print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')

                    head_num_list.append([classHead])

                    # Store ERF mask for next layers especially layers after concatenation
                    # for the same object
                    effective_saliency_mask_per_obj = dict()
                    effective_activation_mask_per_obj = dict()

                    for layer in tqdm.tqdm(self.layer_names):

                        gradients = self.gradients[layer][classHead]
                        activations = self.activations[layer][classHead]

                        activation_map = activations.sum(1, keepdim=True)

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
                        elif self.sel_XAImethod == 'odam':
                            saliency_map = F.relu((gradients * activations).sum(1, keepdim=True))

                        saliency_map_orig = saliency_map.detach()
                        activation_map_orig = activation_map.detach()

                        # # DEBUG: Test ERF with random and sparse saliency map
                        # sparsity= 0.05
                        # saliency_map_orig = torch.rand(saliency_map_orig.shape)
                        # saliency_map_orig = torch.where(torch.logical_and((0.28 < saliency_map_orig), (saliency_map_orig < 0.3)), 
                        #                                 saliency_map_orig, torch.zeros_like(saliency_map_orig))
                        # # saliency_map_orig = torch.bernoulli(torch.full(saliency_map_orig.shape, sparsity))
                        # activation_map_orig = torch.zeros_like(activation_map_orig)

                        saliency_maps_orig_all[layer].append(saliency_map_orig)
                        activation_maps_orig_all[layer].append(activation_map_orig)

                        if saliency_map_orig.max().item() != 0:
                            nObj[layer] = nObj[layer] + 1

                        ## Rescale based on receptive field
                        receptive_field, jump, start = calculate_receptive_field(layer, self.layers)

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

                        theoretical_rf_all[layer] = adjusted_receptive_field

                        if layer in self.layers:
                            receive_input_from = self.layers[layer][1]
                        else:
                            receive_input_from = self.layers[re.sub(r'_cv[12]_act','',layer)][layer][1]

                        if layer == 'model_0_act': # base case
                            effective_saliency_mask_prev = None
                            effective_activation_mask_prev = None
                        else: # if it receive inputs from multiple layers, ERV_prev should be the union
                            effective_saliency_mask_prev = torch.zeros((1,1,h_orig,w_orig))
                            effective_activation_mask_prev = torch.zeros((1,1,h_orig,w_orig))
                            for input_layer in receive_input_from:
                                effective_saliency_mask_prev += effective_saliency_mask_per_obj[input_layer]
                                effective_activation_mask_prev += effective_activation_mask_per_obj[input_layer]
                            effective_saliency_mask_prev = effective_saliency_mask_prev.gt(0).int()
                            effective_activation_mask_prev = effective_activation_mask_prev.gt(0).int()

                        # TODO: Allow taking in bilinear arguments                          
                        if saliency_map_orig.max().item() == 0:
                            saliency_map = dict()
                            for sigma_factor in sigma_factors:
                                saliency_map[sigma_factor] = torch.zeros((1,1,h_orig,w_orig))
                            effective_saliency_mask_per_obj[layer] = torch.zeros((1,1,h_orig,w_orig))   
                            effective_saliency_masks_all[layer].append(torch.zeros((1,1,h_orig,w_orig)))
                            theoretical_saliency_masks_all[layer].append(torch.zeros((1,1,h_orig,w_orig)))
                        else:                            
                            saliency_map, effective_saliency_mask_current, theoretical_saliency_mask_current, effective_saliency_rf, saliency_map_orig_mapped = apply_gaussian_kernel_with_erf(saliency_map_orig, mapped_locs, adjusted_receptive_field, (h_orig,w_orig), 
                                                                                                        effective_mask_prev=effective_saliency_mask_prev, sigma_factors=sigma_factors)
                            saliency_maps_orig_mapped_all[layer].append(saliency_map_orig_mapped)
                            effective_saliency_mask_per_obj[layer] = effective_saliency_mask_current
                            effective_saliency_masks_all[layer].append(effective_saliency_mask_current)
                            theoretical_saliency_masks_all[layer].append(theoretical_saliency_mask_current)
                            effective_saliency_rf_all[layer].append(effective_saliency_rf)
                        
                        if activation_map_orig.max().item() == 0:
                            activation_map = dict()
                            for sigma_factor in sigma_factors:
                                activation_map[sigma_factor] = torch.zeros((1,1,h_orig,w_orig))
                            effective_activation_mask_per_obj[layer] = torch.zeros((1,1,h_orig,w_orig))
                            effective_activation_masks_all[layer].append(torch.zeros((1,1,h_orig,w_orig)))
                            theoretical_activation_masks_all[layer].append(torch.zeros((1,1,h_orig,w_orig)))
                        else:
                            activation_map, effective_activation_mask_current, theoretical_activation_mask_current, effective_activation_rf, activation_map_orig_mapped = apply_gaussian_kernel_with_erf(activation_map_orig, mapped_locs, adjusted_receptive_field, (h_orig,w_orig), 
                                                                                                            effective_mask_prev=effective_activation_mask_prev, sigma_factors=sigma_factors)  
                            activation_maps_orig_mapped_all[layer].append(activation_map_orig_mapped)
                            effective_activation_mask_per_obj[layer] = effective_activation_mask_current
                            effective_activation_masks_all[layer].append(effective_activation_mask_current)
                            theoretical_activation_masks_all[layer].append(theoretical_activation_mask_current)
                            effective_activation_rf_all[layer].append(effective_activation_rf)

                        for sigma_factor in sigma_factors:  
                            if self.sel_norm_str == 'norm' and saliency_map_orig.max().item() != 0:
                                saliency_map[sigma_factor] = (saliency_map[sigma_factor] - saliency_map[sigma_factor].min()).div(saliency_map[sigma_factor].max() - saliency_map[sigma_factor].min()).data
                                activation_map[sigma_factor] = (activation_map[sigma_factor] - activation_map[sigma_factor].min()).div(activation_map[sigma_factor].max() - activation_map[sigma_factor].min()).data

                            if sigma_factor not in saliency_map_sum[layer]:
                                saliency_map_sum[layer][sigma_factor] = saliency_map[sigma_factor]
                                activation_map_sum[layer][sigma_factor] = activation_map[sigma_factor]
                            else:
                                saliency_map_sum[layer][sigma_factor] = saliency_map_sum[layer][sigma_factor] + saliency_map[sigma_factor]
                                activation_map_sum[layer][sigma_factor] = activation_map_sum[layer][sigma_factor] + saliency_map[sigma_factor]

                            # if self.sel_XAImethod == 'odam':
                            saliency_maps[layer][sigma_factor].append(saliency_map[sigma_factor].detach().cpu())
                            activation_maps[layer][sigma_factor].append(activation_map[sigma_factor].detach().cpu())

            for layer in self.layer_names:
            
                for sigma_factor in sigma_factors:

                    if nObj[layer] == 0:
                        saliency_map_sum[layer][sigma_factor] = torch.zeros([1, 1, h_orig, w_orig])
                        saliency_maps[layer][sigma_factor].append(torch.zeros([1, 1, h_orig, w_orig]))

                        activation_map_sum[layer][sigma_factor] = torch.zeros([1, 1, h_orig, w_orig])
                        activation_maps[layer][sigma_factor].append(torch.zeros([1, 1, h_orig, w_orig]))
                    else:
                        saliency_map_sum[layer][sigma_factor] = saliency_map_sum[layer][sigma_factor] / nObj[layer]
                        activation_map_sum[layer][sigma_factor] = activation_map_sum[layer][sigma_factor] / nObj[layer]

                raw_data_rec = []

                if nObj[layer] != 0:
                    self.activations[layer][0] = self.activations[layer][0].detach().cpu()
                    self.activations[layer][1] = self.activations[layer][1].detach().cpu()
                    self.activations[layer][2] = self.activations[layer][2].detach().cpu()
                    pred_logit = pred_logit.detach().cpu()
                    logit = logit.detach().cpu()
                    class_prob = class_prob.detach().cpu()
                    activations = activations.detach().cpu()
                    gradients = gradients.detach().cpu()

        gc.collect()
        torch.cuda.empty_cache()

        FrameStack = []
        # FrameStack = np.empty((len(raw_data_rec),), dtype=np.object)
        # for i in range(len(raw_data_rec)):
        #     FrameStack[i] = raw_data_rec[i]

        # Squeeze, Detach
        # saliency_maps = torch.cat(saliency_maps).squeeze(1)
        # saliency_map_sum = saliency_map_sum.squeeze(0).squeeze(0)
        if len(head_num_list) > 0:
            head_num_list = np.squeeze(head_num_list,1).astype(int)
            class_prob_list = torch.cat(class_prob_list).cpu().detach().numpy()

        return saliency_maps, saliency_map_sum, activation_maps, activation_map_sum,\
                (saliency_maps_orig_all, activation_maps_orig_all),\
                (saliency_maps_orig_mapped_all, activation_maps_orig_mapped_all),\
                (effective_saliency_masks_all, theoretical_saliency_masks_all,effective_activation_masks_all, theoretical_activation_masks_all),\
                (effective_saliency_rf_all,effective_activation_rf_all,theoretical_rf_all),\
                pred_list, class_prob_list, head_num_list, FrameStack

    def __call__(self, input_img, shape_orig, sigma_factors=[-1,2,4]):

        return self.forward(input_img, shape_orig, sigma_factors)
