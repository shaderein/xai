import os,re
import numpy as np
from scipy import io

root_dir = '/mnt/h/jinhan/results/'

def check_single_condition(root_path):    
    model = [m for m in ['faster','yolov5s'] if m in root_path][0]
    dataset = [d for d in ['mscoco','_vehicle','_human'] if d in root_path][0].replace('_','')
    xai_method = [x for x in ['fullgradcamraw','odam'] if x in root_path][0]
    rescale_method = [r for r in ['bilinear','sigma2','sigma4'] if r in root_path][0]

    print(f"{model}_{dataset}_{xai_method}_{rescale_method}")

    if not os.path.exists(root_path): 
        print('NOT STARTED')
        return
    
    skip_images = []
    if dataset == 'mscoco':
        skip_images = ['book_472678','clock_164363','hair drier_178028','hair drier_239041', 'kite_405279', 'mouse_513688', 'toaster_232348', 'toaster_453302', 'toothbrush_218439', 'traffic light_453841']

    layer_count = []

    for dir in os.listdir(root_path):
        if not os.path.isdir(os.path.join(root_path,dir)): continue

        try:
            layer_name = re.findall(r'_[a-zA-Z0-9]+\.[a-zA-Z0-9\.]+_',dir)[0].replace('_','')
        except:
            continue

        if layer_name != 'backbone.res2.0.conv1': return

        layer_count.append(layer_name)

        img_count = []
        empty_count = []
        failed_count = []
        zero_preds = []
        read_error_count = []

        for img_file in os.listdir(os.path.join(root_path,dir)):
            to_skip = [s for s in skip_images if s in img_file]
            if len(to_skip) > 0:
                os.remove(os.path.join(root_path,dir,img_file))
                continue

            if '.mat' not in img_file: continue

            try:
                mat = io.loadmat(os.path.join(root_path,dir,img_file))
            except:
                read_error_count.append(img_file)
                os.remove(os.path.join(root_path,dir,img_file))
                continue

            img_count.append(img_file)

            if mat['masks_ndarray'].sum()==1.5 and mat['masks_ndarray'][0,0]==1 and mat['masks_ndarray'][1,1]==0.5:
                empty_count.append(img_file)
            if isinstance(mat['boxes_pred_xyxy'], np.ndarray) and np.array_equal(mat['boxes_pred_xyxy'], np.array([[0]])): # DEBUG
                zero_preds.append(img_file)
                failed_count.append(img_file)
            elif len(mat['boxes_pred_xyxy']) == 0:
                failed_count.append(img_file)
            # elif len(mat['boxes_pred_xyxy']) > 1:
            #     print(img_file)

        print(f"{layer_name}:\t{len(img_count)} images;\t{len(empty_count)} empty;\t{len(failed_count)} failed;\t{len(zero_preds)} zero instead of []")
        print(f"Failed: {failed_count}")
        if len(read_error_count) > 0:
            print(f"\t{len(read_error_count)} Read error: {read_error_count} Deleted!")
        debug = [img for img in empty_count if img not in failed_count]
        if len(debug) > 0:
            print(f"\tEmpty but not failed: {debug}")

    print(f'{len(layer_count)} layers')

for model in ['faster','yolov5s']:
    for dataset in ['vehicle','human']:
        for upscale_type in ['gaussian_sigma2']:#,'gaussian_sigma4','bilinear']:
            if dataset != 'mscoco':
                xai_path = os.path.join(root_dir,'bdd',f"xai_saliency_maps_{model}_{upscale_type}")
            else:
                xai_path = os.path.join(root_dir,dataset,f"xai_saliency_maps_{model}_{upscale_type}")
            for xai_method in ['fullgradcamraw','odam']:
                if dataset == 'mscoco':
                    path = os.path.join(xai_path,xai_method)
                else:
                    path = os.path.join(xai_path,f"{xai_method}_{dataset}")

                check_single_condition(path)
                print()