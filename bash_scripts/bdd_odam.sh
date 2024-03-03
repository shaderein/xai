#! /bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
conda activate xai-3.8
cd /mnt/h/jinhan/xai/yolov5BDD_samelayer
# cd /mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1

# prob=class
# # bdd
# model=/mnt/h/jinhan/xai/models/yolov5sbdd100k300epoch.pt

# # human

# object=human
# img_path="/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_hum_id_task1009"
# label_path="/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_hum_id_task1009_label"

# method=odam
# script=main_faithful_cal_adaptive_yolo_BDD.py
# output_main_dir="/mnt/h/OneDrive - The University Of Hong Kong/bdd/xai_saliency_maps_same_layer/odam_human"
# echo python -W ignore $script --object $object --model-path $model --method $method --prob $prob --output-main-dir "$output_main_dir" --img-path "$img_path" --label-path "$label_path"

# #vehicle
# object=vehicle
# img_path="/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_veh_id_task0922"
# label_path="/mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_veh_id_task0922_label"

# method=odam
# script=main_faithful_cal_adaptive_yolo_BDD.py
# output_main_dir="/mnt/h/OneDrive - The University Of Hong Kong/bdd/xai_saliency_maps_same_layer/odam_vehicle"
# echo python -W ignore $script --object $object --model-path $model --method $method --prob $prob --output-main-dir "$output_main_dir" --img-path "$img_path" --label-path "$label_path"






###
python -W ignore main_faithful_cal_adaptive_yolo_BDD.py --object human --model-path /mnt/h/jinhan/xai/models/yolov5sbdd100k300epoch.pt --method odam --prob class --output-main-dir "/mnt/h/OneDrive - The University Of Hong Kong/bdd/xai_saliency_maps_same_layer/odam_human" --img-path /mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_hum_id_task1009 --label-path /mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_hum_id_task1009_label
python -W ignore main_faithful_cal_adaptive_yolo_BDD.py --object vehicle --model-path /mnt/h/jinhan/xai/models/yolov5sbdd100k300epoch.pt --method odam --prob class --output-main-dir "/mnt/h/OneDrive - The University Of Hong Kong/bdd/xai_saliency_maps_same_layer/odam_vehicle" --img-path /mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_veh_id_task0922 --label-path /mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1/orib_veh_id_task0922_label