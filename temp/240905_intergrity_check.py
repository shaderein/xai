import os,re, logging
import numpy as np
from scipy import io

logging.basicConfig(filename='./logs/240918_integrity_check_bdd2.log', level=logging.INFO)

root_dir = '/opt/jinhanz/results/'

def check_single_condition(root_path):  
    model = [m for m in ['faster','yolov5s'] if m in root_path][0]
    dataset = [d for d in ['mscoco','_vehicle','_human'] if d in root_path][0].replace('_','')
    xai_method = [x for x in ['fullgradcamraw','odam'] if x in root_path][0]
    rescale_method = [r for r in ['bilinear','sigma2','sigma4'] if r in root_path][0]
    is_act = True if f"_{model}_act_" in root_path else False

    logging.info(f"{model}_{dataset}_{xai_method}_{rescale_method}: {'Activation Map' if is_act else 'Feature Map'}")

    if not os.path.exists(root_path): 
        logging.info('NOT STARTED')
        return
    
    skip_images = []
    if model == 'yolov5s':
        if dataset == 'mscoco':
            skip_images = ['book_472678',"baseball glove_515982","toothbrush_160666","potted plant_473219","bench_350607","truck_295420","toaster_232348","kite_405279","toothbrush_218439","snowboard_425906","car_227511","traffic light_453841","hair drier_239041","hair drier_178028","toaster_453302","mouse_513688","spoon_88040","scissors_340930","handbag_383842"]
        elif dataset == 'vehicle':
            skip_images = ["178", "54", "452", "478", "629", "758", "856",'1007', '1028', '1041', '1065', '1100', '1149', '1236', '1258', '1272', '1331', '1356', '210', '222', '3', '390', '431', '485', '505', '52', '559', '585', '634', '648', '670', '715', '784', '797', '803', '833', '848', '867', '899', '914', '940', '980', '993','1121', '1127', '1170', '1365', '321', '425', '542', '610', '896', '902', '953', '967']
        elif dataset == 'human':
            skip_images = ['1022', '1041', '1053', '1063', '1066', '1097', '11', '1141', '1142', '1154', '1227', '1228', '1273', '1293', '1302', '1313', '1346', '1359', '1398', '1420', '1430', '1475', '1506', '152', '1538', '1553', '1624', '1663', '1664', '1746', '1770', '1788', '1803', '1805', '1817', '1852', '186', '1863', '1893', '19', '1917', '1954', '2008', '2040', '2087', '2092', '2108', '2121', '2128', '2141', '2161', '2186', '2203', '2219', '2226', '2262', '2270', '2271', '2279', '231', '2312', '2327', '2334', '2457', '250', '286', '348', '388', '391', '415', '422', '425', '452', '47', '608', '670', '683', '748', '757', '805', '808', '829', '845', '85', '875', '897', '900', '928', '962', '97', '997']

    layer_count = []

    for dir in os.listdir(root_path):
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
            to_skip = [s for s in skip_images if s == img_file.split('-')[0]]
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
    for dataset in ['vehicle','human']:
        for upscale_type in ['bilinear','gaussian_sigma2','gaussian_sigma4']:
            for is_act in ['','_act']:

                if dataset != 'mscoco':
                    xai_path = os.path.join(root_dir,'bdd',f"xai_saliency_maps_{model}{is_act}_{upscale_type}")
                else:
                    xai_path = os.path.join(root_dir,dataset,f"xai_saliency_maps_{model}{is_act}_{upscale_type}")
                for xai_method in ['odam','fullgradcamraw']:
                    if dataset == 'mscoco':
                        path = os.path.join(xai_path,xai_method)
                    else:
                        path = os.path.join(xai_path,f"{xai_method}_{dataset}")

                    if xai_method == 'fullgradcamraw':
                        path = path.replace(root_dir,f"{root_dir}/no_mmn/")

                    check_single_condition(path)