# from matplotlib import pyplot as plt
import os, re, shutil

import cv2

xai_saliency_maps_path = "/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/multi_layer_analysis/odam_test_results"
# xai_saliency_maps_path = "/mnt/h/jinhan/xai/yolov5BDD/multi_layer_analysis/odam_test_results"

for root, dirs, files in os.walk(xai_saliency_maps_path):
    for dir in dirs:
        layer_num = re.findall(r"F\d+",dir)
        file = os.path.join(root, dir, '1079-res.jpg')
        # file = os.path.join(root, dir, '148-res.jpg')
        target_file = f"/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/multi_layer_analysis/odam_test_results/1076-xai-res-{layer_num}.jpg"
        # target_file = f"/mnt/h/jinhan/xai/yolov5BDD/multi_layer_analysis/odam_test_results/148-xai-res-{layer_num}.jpg"
        shutil.copy(file, target_file)

        img = cv2.imread(target_file)
        # crop_img = img[:,round(img.shape[1]/3):]
        # cv2.imwrite(target_file, crop_img)