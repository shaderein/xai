import time
import torch
import torch.nn.functional as F
import gc
import utils.util_my_yolov5 as ut
import numpy as np
from collections import defaultdict

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
            kernel_size, layer_stride, layer_padding = list(layers[layer].values())[0]
        else:
            kernel_size, layer_stride, layer_padding = layers[layer]

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

def map_saliency_to_original_image(saliency_height, saliency_width, original_size, preprocessed_size, 
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

    return mapped_locs

def create_gaussian_kernel(size, sigma):
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    y = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize the kernel
    return kernel

def apply_gaussian_kernel_torch(saliency_map, mapped_locs, receptive_field, original_size, sigma_factor=2):
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
    output = torch.zeros((N, C, original_height, original_width), dtype=np.float32, device=saliency_map.device) #TODO

    for n in range(N):
        # debug
        sigma_x = receptive_field[n, 0] / sigma_factor
        sigma_y = receptive_field[n, 1] / sigma_factor

        kernel_size_x = math.ceil(sigma_x * 6)
        kernel_size_y = math.ceil(sigma_y * 6)
        if kernel_size_x % 2 == 0: kernel_size_x += 1
        if kernel_size_y % 2 == 0: kernel_size_y += 1

        gaussian_x = cv2.getGaussianKernel(ksize=kernel_size_x, sigma=sigma_x)
        gaussian_y = cv2.getGaussianKernel(ksize=kernel_size_y, sigma=sigma_y)
        gaussian = gaussian_y * gaussian_x.T  # Create 2D Gaussian kernel
        h, w = gaussian.shape

        center_x = mapped_locs[n, :, :, 0].astype(int)
        center_y = mapped_locs[n, :, :, 1].astype(int)

        # Determine the region in the output array
        sx = np.maximum(center_x - w // 2, 0)
        ex = np.minimum(center_x + w // 2 + 1, original_width)
        sy = np.maximum(center_y - h // 2, 0)
        ey = np.minimum(center_y + h // 2 + 1, original_height)

        # Determine the relative region in the gaussian kernel
        gsx = np.maximum(0, w // 2 - (center_x - sx))
        gex = gsx + (ex - sx)
        gsy = np.maximum(0, h // 2 - (center_y - sy))
        gey = gsy + (ey - sy)

        for i in range(saliency_height):
            for j in range(saliency_width):
                cx, cy = center_x[i, j], center_y[i, j]
                if 0 <= cx < original_width and 0 <= cy < original_height:
                    output[n, 0, sy[i, j]:ey[i, j], sx[i, j]:ex[i, j]] += (
                        saliency_map[n, 0, i, j] * gaussian[gsy[i, j]:gey[i, j], gsx[i, j]:gex[i, j]]
                    )

    return torch.tensor(output)

def apply_gaussian_kernel(saliency_map, mapped_locs, sigma, original_size):
    """
    Apply Gaussian kernel to the saliency map locations and upsample to the original image size.
    Args:
    - saliency_map (N,C,H,W): N=#proposals for head and N=1 for backone. C=1 (summed over channls in fullgradcam)
    - mapped_locs (N,H,W,2): Mapped locations on the original image.
    - original_size (tuple): The size of the original image.
    - sigmas (N,2): Sigma value at x,y dimension

    Returns:
    - output (ndarray): Saliency map upsampled to the original image size.
    """
    N, C, saliency_height, saliency_width = saliency_map.shape
    original_height, original_width = original_size
    output = np.zeros((N, C, original_height, original_width), dtype=np.float32) #TODO

    saliency_map = saliency_map.cpu().detach().numpy()

    for n in range(N):
        # debug
        sigma_x = sigma # TODO: fasterRCNN each sigma for each proposal?
        sigma_y = sigma

        kernel_size_x = math.ceil(sigma_x * 6)
        kernel_size_y = math.ceil(sigma_y * 6)
        if kernel_size_x % 2 == 0: kernel_size_x += 1
        if kernel_size_y % 2 == 0: kernel_size_y += 1

        gaussian_x = cv2.getGaussianKernel(ksize=kernel_size_x, sigma=sigma_x)
        gaussian_y = cv2.getGaussianKernel(ksize=kernel_size_y, sigma=sigma_y)
        gaussian = gaussian_y * gaussian_x.T  # Create 2D Gaussian kernel
        h, w = gaussian.shape

        center_x = mapped_locs[n, :, :, 0].astype(int)
        center_y = mapped_locs[n, :, :, 1].astype(int)

        # Determine the region in the output array
        sx = np.maximum(center_x - w // 2, 0)
        ex = np.minimum(center_x + w // 2 + 1, original_width)
        sy = np.maximum(center_y - h // 2, 0)
        ey = np.minimum(center_y + h // 2 + 1, original_height)

        # Determine the relative region in the gaussian kernel
        gsx = np.maximum(0, w // 2 - (center_x - sx))
        gex = gsx + (ex - sx)
        gsy = np.maximum(0, h // 2 - (center_y - sy))
        gey = gsy + (ey - sy)

        for i in range(saliency_height):
            for j in range(saliency_width):
                cx, cy = center_x[i, j], center_y[i, j]
                if 0 <= cx < original_width and 0 <= cy < original_height:
                    output[n, 0, sy[i, j]:ey[i, j], sx[i, j]:ex[i, j]] += (
                        saliency_map[n, 0, i, j] * gaussian[gsy[i, j]:gey[i, j], gsx[i, j]:gex[i, j]]
                    )

    return torch.tensor(output)

def apply_gaussian_kernel_gpu(saliency_map, mapped_locs, sigma, original_size):
    """
    Apply Gaussian kernel to the saliency map locations and upsample to the original image size using GPU.
    Args:
    - saliency_map (N,C,H,W): N=#proposals for head and N=1 for backbone. C=1 (summed over channels in fullgradcam)
    - mapped_locs (N,H,W,2): Mapped locations on the original image.
    - original_size (tuple): The size of the original image.
    - sigma (float): Sigma value for the Gaussian kernel.

    Returns:
    - output (torch.Tensor): Saliency map upsampled to the original image size.
    """
    mapped_locs = torch.tensor(mapped_locs)
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

    # Apply Gaussian kernel for all points at once (vectorized)
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


class YOLOV5XAI:

    def __init__(self, model, layer_names, sel_prob_str, sel_norm_str, sel_classes, sel_XAImethod,layers, sigma_factors=[-1,2,4],device=0,img_size=(608, 608),):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        self.sel_prob_str = sel_prob_str
        self.sel_norm_str = sel_norm_str
        self.sel_classes = sel_classes
        self.sel_XAImethod = sel_XAImethod
        self.layer_name = layer_names[0]
        self.layers = layers # layer info
        self.sigma_factors = sigma_factors
        self.sel_device = device

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
        device = f'cuda:{self.sel_device}' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        # print('[INFO] saliency_map size :', self.activations[0].shape[2:])

        target_layer = find_yolo_layer(self.model, layer_names[1])
        target_layer.register_forward_hook(forward_hook_1)
        target_layer.register_backward_hook(backward_hook_1)
        device = f'cuda:{self.sel_device}' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        # print('[INFO] saliency_map size :', self.activations[1].shape[2:])

        target_layer = find_yolo_layer(self.model, layer_names[2])
        target_layer.register_forward_hook(forward_hook_2)
        target_layer.register_backward_hook(backward_hook_2)
        device = f'cuda:{self.sel_device}' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        # print('[INFO] saliency_map size :', self.activations[2].shape[2:])

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
        saliency_maps_orig_all = []
        activation_maps_orig_all = []

        class_prob_list = []
        head_num_list = []
        nObj = 0
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

        ## Rescale based on receptive field
        receptive_field, jump, start = calculate_receptive_field(self.layer_name, self.layers)

        # Adjust the receptive field to account for the input resizing in preprocessing
        height_ratio = h_orig / h
        width_ratio = w_orig / w
        adjusted_receptive_field = receptive_field
        adjusted_receptive_field[:,0] = receptive_field[:,0] * width_ratio
        adjusted_receptive_field[:,1] = receptive_field[:,1] * height_ratio

        # Get a grid of coordinates of the receptive field center of each spatial location of the intermediate saliency map, 
        #   mapped to the original image
        mapped_locs = map_saliency_to_original_image(self.activations[0].shape[2], self.activations[0].shape[3], (h_orig,w_orig), (h,w), jump, start) #DEBUG: find the head always with output (shortest path)

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
                    gradients = self.gradients[classHead]
                    activations = self.activations[classHead]

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
                    elif self.sel_XAImethod == 'saveRawAllAct':
                        saliency_map = ut.gradcampp_operation(activations, gradients)
                    elif self.sel_XAImethod == 'DRISE':
                        res = self.generate_saliency_map(input_img, bbox, prob_thresh=0.5, grid_size=(16, 16), n_masks=5000, seed=0)
                        res_expanded = np.expand_dims(np.expand_dims(res, axis=0), axis=0)
                        saliency_map = torch.tensor(res_expanded, device=input_img.device)
                    elif self.sel_XAImethod == 'odam':
                        saliency_map = F.relu((gradients * activations).sum(1, keepdim=True))

                    saliency_map_orig = saliency_map.detach().cpu()
                    activation_map_orig = activation_map.detach().cpu()

                    saliency_maps_orig_all.append(saliency_map_orig)
                    activation_maps_orig_all.append(activation_map_orig)

            raw_data_rec = []

            if nObj != 0:
                self.activations[0] = self.activations[0].detach().cpu()
                self.activations[1] = self.activations[1].detach().cpu()
                self.activations[2] = self.activations[2].detach().cpu()
                pred_logit = pred_logit.detach().cpu()
                logit = logit.detach().cpu()
                class_prob = class_prob.detach().cpu()
                activations = activations.detach().cpu()
                gradients = gradients.detach().cpu()

        gc.collect()
        torch.cuda.empty_cache()

        FrameStack = []

        if len(head_num_list) > 0:
            head_num_list = np.squeeze(head_num_list,1).astype(int)
            class_prob_list = torch.cat(class_prob_list).cpu().detach().numpy()

        return (saliency_maps_orig_all, activation_maps_orig_all),\
                mapped_locs, adjusted_receptive_field,\
                pred_list, class_prob_list, head_num_list, FrameStack

    def __call__(self, input_img, shape_orig):

        return self.forward(input_img, shape_orig)
