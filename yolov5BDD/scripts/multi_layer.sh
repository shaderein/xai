#! /bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
prob=class
method=fullgradcamraw

# MSCOCO
conda activate xai-coco
cd /mnt/h/jinhan/xai/yolov5COCO
model_path="/mnt/h/jinhan/xai/models/yolov5s_COCOPretrained.pt"

output_main_dir="/mnt/h/jinhan/results/mscoco/xai_saliency_maps_yolov5s/fullgradcamraw_by_head"
object="COCO"
coco_labels="/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/COCO_classes2.txt"
img_path="/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/DET2"
label_path="/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/annotations_DET2"
# echo "python -W ignore main_faithful_cal_adaptive_yolo_COCO_by_head.py --output-main-dir "$output_main_dir" --model-path "$model_path" --object "$object" --method "$method" --prob $prob --img-path "$img_path" --label-path "$label_path""
python -W ignore main_faithful_cal_adaptive_yolo_COCO_by_head.py --coco-labels "$coco_labels" --output-main-dir "$output_main_dir" --model-path "$model_path" --object "$object" --method "$method" --prob $prob --img-path "$img_path" --label-path "$label_path"

# BDD
conda activate xai-3.8
cd /mnt/h/jinhan/xai/yolov5BDD
model_path="/mnt/h/jinhan/xai/models/yolov5sbdd100k300epoch.pt"

# Human
output_main_dir="/mnt/h/jinhan/results/bdd/xai_saliency_maps_yolov5s/human_fullgradcamraw_by_head"
object="human"
img_path="/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_hum_id_task1009"
label_path="/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_hum_id_task1009_label"
# echo "python -W ignore main_faithful_cal_adaptive_yolo_BDD_by_head.py --output-main-dir "$output_main_dir" --model-path "$model_path" --object "$object" --method "$method" --prob $prob --img-path "$img_path" --label-path "$label_path""
python -W ignore main_faithful_cal_adaptive_yolo_BDD_by_head.py --output-main-dir "$output_main_dir" --model-path "$model_path" --object "$object" --method "$method" --prob $prob --img-path "$img_path" --label-path "$label_path"

# Vehicle
output_main_dir="/mnt/h/jinhan/results/bdd/xai_saliency_maps_yolov5s/vehicle_fullgradcamraw_by_head"
object="vehicle"
img_path="/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_veh_id_task0922"
label_path="/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_veh_id_task0922_label"
# echo "python -W ignore main_faithful_cal_adaptive_yolo_BDD_by_head.py --output-main-dir "$output_main_dir" --model-path "$model_path" --object "$object" --method "$method" --prob $prob --img-path "$img_path" --label-path "$label_path""
python -W ignore main_faithful_cal_adaptive_yolo_BDD_by_head.py --output-main-dir "$output_main_dir" --model-path "$model_path" --object "$object" --method "$method" --prob $prob --img-path "$img_path" --label-path "$label_path"

