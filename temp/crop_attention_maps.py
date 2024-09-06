import matplotlib.pyplot as plt
import os, re, torch,cv2
import scipy.io
from collections import defaultdict
import numpy as np
import pandas as pd

# from ..yolov5BDD.utils.util_my_yolov5 import load_gt_labels
# TODO: use relative import
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def concat_images(images):
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        base_img[:, h * i:h * (i + 1), ...] = img
    return base_img

def load_gt_labels(img, label_path, class_names_gt, class_names_sel):
    label_data = np.loadtxt(label_path, dtype=np.float32, delimiter=' ')
    if len(label_data.shape) == 1:
        label_data = label_data[None,:]
    label_data_class = label_data[:, 0]
    label_data_corr = label_data[:,1:]
    sel_idx = []
    for class_name_sel in class_names_sel:
        sel_idx.append(class_names_gt.index(class_name_sel))
    sel_bbox_idx = []
    label_data_class_names = []
    for i, i_label_data_class in enumerate(label_data_class):
        if i_label_data_class in sel_idx:
            sel_bbox_idx.append(i)
            label_data_class_names.append(class_names_gt[i_label_data_class.astype('int32')])
    label_data_corr = label_data_corr[sel_bbox_idx,:] #filter classes
    label_data_class = label_data_class[sel_bbox_idx] #filter class labels
    img_h, img_w = np.size(img, 0), np.size(img, 1)
    label_data_corr[:, 0] = label_data_corr[:, 0] * img_w
    label_data_corr[:, 1] = label_data_corr[:, 1] * img_h
    label_data_corr[:, 2] = label_data_corr[:, 2] * img_w
    label_data_corr[:, 3] = label_data_corr[:, 3] * img_h
    label_data_corr_xywh = label_data_corr
    label_data_corr = xywh2xyxy(label_data_corr)
    label_data_corr_xyxy = label_data_corr
    label_data_corr_yxyx = label_data_corr[:, [1,0,3,2]]
    label_data_corr_yxyx = np.round(label_data_corr_yxyx)
    boxes_GT = label_data_corr_yxyx[:, None, :].tolist()

    return boxes_GT, label_data_corr_xyxy, label_data_corr_xywh, label_data_corr_yxyx, label_data_class, label_data_class_names

def get_res_img(mask, res_img):
    mask = torch.from_numpy(mask)
    mask = mask.mul(255).add_(0.5).clamp_(0, 255).numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = (heatmap/255).astype(np.float32)
    #n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    res_img = (res_img / 255).astype(np.float32)
    res_img = cv2.add(res_img, heatmap)
    res_img = (res_img / res_img.max())
    return res_img, heatmap

def crop_attention_map(map, bb_yxyx):
    y1,x1,y2,x2 = bb_yxyx
    y1,x1,y2,x2 = int(y1),int(x1),int(y2),int(x2)

    ## x1 = -1
    x1 = max(0,x1)
    # x2 = min(x2, 1024)
    y1 = min(0,y1)
    # y2 = max(y2,576)

    filtered_array = np.zeros(map.shape)

    filtered_array[y1:y2 + 1, x1:x2 + 1] = map[y1:y2 + 1, x1:x2 + 1]
    
    # for row in range(y1, y2 + 1):
    #     for col in range(x1, x2 + 1):
    #         filtered[row][col] = map[row][col]

    return filtered_array

# Select BB used in EXP
bb_selections = {
    'vehicle': pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/bdd/labels_mapping/Random_sample_vehicle_procedure_analysis.xlsx','veh_sample_img_condition')[['image','vehicle_count_gt','ExpTargetIndex']],
    'human': pd.read_excel('/mnt/h/OneDrive - The University Of Hong Kong/bdd/labels_mapping/Random_sample_human_procedure_analysis.xlsx','hum_sample_img_condition')[['imgnumber','human_count_gt','ExpTargetIndex']]
}

bb_annotations_path = {
    'vehicle': '/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_veh_id_task0922_label',
    'human': '/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_hum_id_task1009_label'
}

bb_annotations = defaultdict(defaultdict)

img = np.zeros([576,1024])
class_names_gt = ['person', 'rider', 'car', 'bus', 'truck']

for category, path in bb_annotations_path.items():
    if category == 'vehicle': class_names_sel = ['car', 'bus', 'truck']
    elif category == "human": class_names_sel = ['person', 'rider']

    for file in os.listdir(path):
        img_idx = int(file.replace('.txt',''))

        if category=='vehicle':
            bb_selection = bb_selections[category].loc[bb_selections[category]['image']==f"{img_idx}.jpg"]
        elif category=='human':
            bb_selection = bb_selections[category].loc[bb_selections[category]['imgnumber']==img_idx]
        
        boxes_GT, label_data_corr_xyxy, label_data_corr_xywh, label_data_corr_yxyx, label_data_class, label_data_class_names\
        = load_gt_labels(img, os.path.join(path,file), class_names_gt, class_names_sel)

        indices_GT_sorted = np.concatenate(boxes_GT,axis=0)[:, 1].argsort() # use x1 (top-left) to determine order of target (left to right)
        target_idx_GT = indices_GT_sorted[bb_selection['ExpTargetIndex'].values[0]-1]

        bb_annotations[category][img_idx] = boxes_GT[target_idx_GT][0] # [[y1 x1 y2 x2]]

# Whole-image Attention Maps
human_attention_path = {
    'vehicle': '/mnt/h/OneDrive - The University Of Hong Kong/bdd/attention_maps/231206 Veh DET/whole_image',
    'human':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/attention_maps/231206 Hum DET/whole_image',
}

cropped_attention_path = {
    'vehicle': '/mnt/h/OneDrive - The University Of Hong Kong/bdd/attention_maps/231206 Veh DET/cropped',
    'human':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/attention_maps/231206 Hum DET/cropped',
}


images_path = {
    'vehicle':'/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_veh_id_task0922',
    'human':'/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_hum_id_task1009'
}

plot_path = {
    'vehicle': '/mnt/h/OneDrive - The University Of Hong Kong/bdd/attention_maps/231206 Veh DET/visualize',
    'human':'/mnt/h/OneDrive - The University Of Hong Kong/bdd/attention_maps/231206 Hum DET/visualize',
}

human_attention = defaultdict(defaultdict)
cropped_attention = defaultdict(defaultdict)

for category, path in human_attention_path.items():
    for file in os.listdir(path):
        img_idx = int(re.findall(r'\d+_',file)[-1].replace('_',''))
        mat = scipy.io.loadmat(os.path.join(path,file))

        map = mat['output_map_norm']
        cropped_map = crop_attention_map(map, bb_annotations[category][img_idx])
        
        human_attention[category][img_idx] = map
        cropped_attention[category][img_idx] = cropped_map

        # Save
        scipy.io.savemat(os.path.join(cropped_attention_path[category],file.replace('.mat','_cropped.mat')),
                         mdict={'output_map_norm':cropped_map})
        # np.save(os.path.join(cropped_attention_path[category],f"{img_idx}.npy"),cropped_map)

        # Visualize
        img_path = os.path.join(images_path[category], f'{img_idx}.jpg')
        img = cv2.imread(img_path)

        res_img_whole, _ = get_res_img(map, img.copy())
        res_img_cropped, _ = get_res_img(cropped_map, img.copy())

        final_image = concat_images([res_img_whole*255, res_img_cropped*255])
        cv2.imwrite(os.path.join(plot_path[category],f"{img_idx}_cropped.jpg"), final_image)
