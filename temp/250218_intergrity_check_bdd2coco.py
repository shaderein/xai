import os,re, logging, torch
import numpy as np
from scipy import io

logging.basicConfig(filename='./logs/250218_integrity_check_bdd2.log', level=logging.INFO)

root_dir = '/opt/jinhanz/results/'

def check_single_condition(root_path):  
    model = [m for m in ['faster','yolov5s'] if m in root_path][0]
    dataset = [d for d in ['mscoco','_vehicle','_human'] if d in root_path][0].replace('_','')
    xai_method = [x for x in ['fullgradcamraw','odam'] if x in root_path][0]
    rescale_method = [r for r in ['bilinear','optimize_faithfulness_finer_v2.5'] if r in root_path][0]
    is_act = True if f"activation_map" in root_path else False

    logging.info(f"{model}_{dataset}_{xai_method}_{rescale_method}: {'Activation Map' if is_act else 'Feature Map'}")

    if not os.path.exists(root_path): 
        logging.info('NOT STARTED')
        return
    
    layer_count = []

    for dir in os.listdir(root_path):
        if '.pth' in dir: continue
        if not os.path.isdir(os.path.join(root_path,dir)): continue

        if model == 'faster':
            try:
                layer_name = re.findall(r'_[a-zA-Z0-9]+\.[a-zA-Z0-9\.]+_',dir)[0].replace('_','')
            except:
                continue
        elif model == 'yolov5s':
            try:
                layer_name = re.findall(r'_model_[a-zA-Z0-9_]+_act_',dir)[0][1:-1]
            except:
                continue

        if any([l for l in ['model_18','model_20','model_21','model_23'] if l in layer_name]): continue

        layer_count.append(layer_name)

        img_count = []
        empty_count = []
        failed_count = []
        zero_preds = []
        read_error_count = []

        for img_file in os.listdir(os.path.join(root_path,dir)):

            if '.pth' not in img_file: continue

            try:
                mat = torch.load(os.path.join(root_path,dir,img_file))
            except:
                read_error_count.append(img_file)
                os.remove(os.path.join(root_path,dir,img_file))
                continue

            img_count.append(img_file)

            if mat['masks_ndarray'].sum()==1.5 and mat['masks_ndarray'][0,0]==1 and mat['masks_ndarray'][1,1]==0.5:
                empty_count.append(img_file)
            # if isinstance(mat['boxes_pred_xyxy'], np.ndarray) and np.array_equal(mat['boxes_pred_xyxy'], np.array([[0]])): # DEBUG
            #     zero_preds.append(img_file)
            #     failed_count.append(img_file)
            # elif len(mat['boxes_pred_xyxy']) == 0:
            #     failed_count.append(img_file)
            # elif len(mat['boxes_pred_xyxy']) > 1:
            #     logging.info(img_file)

        logging.info(f"{layer_name}:\t{len(img_count)} images;\t{len(empty_count)} empty;\t{len(failed_count)} failed;\t{len(zero_preds)} zero instead of []")
        if len(empty_count) > 0:
            logging.warning(f"Empty: {empty_count}")
        if len(failed_count) > 0:
            logging.warning(f"Failed: {failed_count}")
        if len(read_error_count) > 0:
            logging.warning(f"\t{len(read_error_count)} Read error: {read_error_count} Deleted!")
        debug = [img for img in empty_count if img not in failed_count]
        if len(debug) > 0:
            logging.warning(f"\tEmpty but not failed: {debug}")

    logging.info(f'{len(layer_count)} layers')

for model in ['yolov5s']:
    for dataset in ['human']:
        for upscale_type in ['optimize_faithfulness_finer_v2.5','bilinear']:
            for is_act in ['xai_saliency','activation']:

                xai_path = os.path.join(root_dir,model,upscale_type,'bdd2coco',f"{is_act}_maps_{model}")

                for xai_method in ['fullgradcamraw','odam']:
                    if dataset == 'mscoco':
                        path = os.path.join(xai_path,xai_method)
                    else:
                        path = os.path.join(xai_path,f"{xai_method}_{dataset}")

                    check_single_condition(path)