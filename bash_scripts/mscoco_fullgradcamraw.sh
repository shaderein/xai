#! /bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
conda activate xai-coco
cd /mnt/h/jinhan/xai/yolov5COCO_guoyang_newlayer
# cd /mnt/h/Projects/HKU_XAI_Project/Yolov5self_GradCAM_Pytorch_1

# prob=class
# # mscoco
# model=/mnt/h/jinhan/xai/models/yolov5s_COCOPretrained.pt

# object=COCO
# img_path="/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/DET"
# label_path="/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/annotations_DET"
# coco_labels="/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/COCO_classes.txt"

# method=odam
# script=main_faithful_cal_adaptive_yolo_COCO.py
# output_main_dir="/mnt/h/OneDrive - The University Of Hong Kong/mscoco/xai_saliency_maps/odam"
# echo python -W ignore $script --object $object --model-path $model --method $method --prob $prob --output-main-dir "$output_main_dir" --coco-labels "$coco_labels" --img-path "$img_path" --label-path "$label_path"

# method=fullgradcamraw
# script=main_faithful_cal_adaptive_yolo_COCO_whole_image.py
# output_main_dir="/mnt/h/OneDrive - The University Of Hong Kong/mscoco/xai_saliency_maps/fullgradcamraw"
# echo python -W ignore $script --object $object --model-path $model --method $method --prob $prob --output-main-dir "$output_main_dir" --coco-labels "$coco_labels" --img-path "$img_path" --label-path "$label_path"

# python -W ignore main_faithful_cal_adaptive_yolo_COCO.py --object COCO --model-path /mnt/h/jinhan/xai/models/yolov5s_COCOPretrained.pt --method odam --prob class --output-main-dir "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/xai_saliency_maps/odam" --coco-labels "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/COCO_classes.txt" --img-path "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/DET" --label-path "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/annotations_DET"
python -W ignore main_faithful_cal_adaptive_yolo_COCO_whole_image.py --object COCO --model-path /mnt/h/jinhan/xai/models/yolov5s_COCOPretrained.pt --method fullgradcamraw --prob class --output-main-dir "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/xai_saliency_maps/fullgradcamraw" --coco-labels "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/COCO_classes2.txt" --img-path "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/images/resized/DET2" --label-path "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/annotations/annotations_DET2"