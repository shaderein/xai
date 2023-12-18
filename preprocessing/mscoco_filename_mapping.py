import os, re, shutil
import pandas as pd

official_img_path = '/mnt/h/Projects/HKU_XAI_Project/Yolov5pretrained_GradCAM_Pytorch_1/COCO_YOLO_IMAGE'
official_annotation_path = '/mnt/h/Projects/HKU_XAI_Project/Yolov5pretrained_GradCAM_Pytorch_1/COCO_YOLO_LABEL'

experiment_img_table_path = '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/other/target_boxes_for_eyegaze_explanation_new_xywh.xlsx'

target_img_path = '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/Raw_Images'
target_annotation_path = '/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/annotations_DET'

experiment = pd.read_excel(experiment_img_table_path)

for img in experiment.iloc[:,0]:
    idx = re.findall(r"\d+",img)[0]
    shutil.copyfile(os.path.join(official_img_path,f"{idx.zfill(12)}.jpg"),
                    os.path.join(target_img_path, img.replace('.png','.jpg')))
    shutil.copyfile(os.path.join(official_annotation_path,f"{idx.zfill(12)}.txt"),
                    os.path.join(target_annotation_path, img.replace('.png','.txt')))
