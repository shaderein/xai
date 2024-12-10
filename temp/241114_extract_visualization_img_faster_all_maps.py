import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os, re, shutil, torch, tqdm

import numpy as np

import cv2

def get_res_img(mask, res_img):
    mask = mask.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = (heatmap/255).astype(np.float32)
    #n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    res_img = (res_img / 255).astype(np.float32)
    res_img = cv2.add(res_img, heatmap)
    res_img = (res_img / res_img.max())
    return res_img, heatmap

def combine_images(is_act,rescale_method,root_dir, target_img_name, ext):
    dataset = 'vehicle'
    model = 'FasterRCNN_C4_BDD100K_1'
    if 'mscoco' in root_dir:
        dataset = 'COCO'
        model = 'model_final_721ade_1'
    elif 'vehicle' in root_dir:
        dataset = 'vehicle'
        model = 'FasterRCNN_C4_BDD100K_1'
    elif 'human' in root_dir:
        dataset = 'human'
        model = 'FasterRCNN_C4_BDD100K_1'

    saved_path = f"results/visualizations/241114_faster_all_layers/{dataset}/{rescale_method}_{is_act}_{target_img_name}.png"

    if os.path.exists(saved_path): return

    title_font_size = 15

    layers = ['backbone.res2.0.conv1','backbone.res2.0.conv2','backbone.res2.0.conv3','backbone.res2.1.conv1','backbone.res2.1.conv2','backbone.res2.1.conv3','backbone.res2.2.conv1','backbone.res2.2.conv2','backbone.res2.2.conv3','backbone.res3.0.conv1','backbone.res3.0.conv2','backbone.res3.0.conv3','backbone.res3.1.conv1','backbone.res3.1.conv2','backbone.res3.1.conv3','backbone.res3.2.conv1','backbone.res3.2.conv2','backbone.res3.2.conv3','backbone.res3.3.conv1','backbone.res3.3.conv2','backbone.res3.3.conv3','backbone.res4.0.conv1','backbone.res4.0.conv2','backbone.res4.0.conv3','backbone.res4.1.conv1','backbone.res4.1.conv2','backbone.res4.1.conv3','backbone.res4.2.conv1','backbone.res4.2.conv2','backbone.res4.2.conv3','backbone.res4.3.conv1','backbone.res4.3.conv2','backbone.res4.3.conv3','backbone.res4.4.conv1','backbone.res4.4.conv2','backbone.res4.4.conv3','backbone.res4.5.conv1','backbone.res4.5.conv2','backbone.res4.5.conv3','roi_heads.pooler.level_poolers.0','roi_heads.res5.0.conv1','roi_heads.res5.0.conv2','roi_heads.res5.0.conv3','roi_heads.res5.1.conv1','roi_heads.res5.1.conv2','roi_heads.res5.1.conv3','roi_heads.res5.2.conv1','roi_heads.res5.2.conv2','roi_heads.res5.2.conv3']

    rows = 5
    cols = 10
    fig, axes = plt.subplots(rows,cols,figsize=(40,20))

    layer_idx = 1

    if dataset == 'COCO':
        result = cv2.imread(f"/home/jinhanz/cs/data/mscoco/images/resized/DET2/{target_img_name}.png")
    elif dataset == 'vehicle':
        result = cv2.imread(f"/home/jinhanz/cs/data/bdd/orib_veh_id_task0922/{target_img_name}.jpg")
       

    for i in tqdm.tqdm(range(rows)):
        for j in range(cols):
            if layer_idx > len(layers): break
            
            layer = layers[layer_idx-1]

            row = i

            if dataset == 'COCO':
                fullgradcam_saliency_mask = torch.as_tensor(torch.load(os.path.join(root_dir,"fullgradcamraw",f"fullgradcamraw_{dataset}_NMS_class_{layer}_aifaith_norm_{model}",f"{target_img_name}{ext}.pth"))['masks_ndarray'])
            else:
                fullgradcam_saliency_mask = torch.as_tensor(torch.load(os.path.join(root_dir,f"fullgradcamraw_{dataset}",f"fullgradcamraw_{dataset}_NMS_class_{layer}_aifaith_norm_{model}",f"{target_img_name}{ext}.pth"))['masks_ndarray'])

            res_img = result.copy()
            res_img, _ = get_res_img(fullgradcam_saliency_mask.unsqueeze(0), res_img)
            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            axes[row,j].imshow(res_img, extent=[0,result.shape[1],result.shape[0],0])

            # odam_saliency_mask = torch.as_tensor(torch.load(os.path.join(root_dir,"odam",f"odam_COCO_NMS_class_{layer}_aifaith_norm_model_final_721ade_1",f"{target_img_name}{ext}.pth"))['masks_ndarray'])

            # res_img = result.copy()
            # res_img, _ = get_res_img(odam_saliency_mask.unsqueeze(0), res_img)
            # res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            # axes[row+1,j].imshow(res_img, extent=[0,result.shape[1],result.shape[0],0])
            
            if "pooler" in layer:
                axes[row,j].set_title(f"[{layer_idx}] {layer}\nFullGradCAM",fontsize=int(title_font_size*0.8))
            else:
                axes[row,j].set_title(f"[{layer_idx}] {layer}\nFullGradCAM",fontsize=title_font_size)
                # axes[row+1,j].set_title("ODAM",fontsize=title_font_size)
            axes[row,j].axis('off')
            # axes[row+1,j].axis('off')
            layer_idx += 1

        #Hide any unused axes in the current row
        # for j in range(len(layers[i]), cols):
    fig.delaxes(axes[4, 9])

    plt.tight_layout()
    plt.savefig(saved_path)

# root_dir = "/mnt/h/OneDrive - The University Of Hong Kong/mscoco/xai_saliency_maps_faster/fullgradcamraw"
# target_img_name = 'chair_81061'
# ext = 'png'
# combine_images(root_dir,target_img_name,ext)

# for sigma in ['bilinear','gaussian_sigma2','gaussian_sigma4']:
#     root_dir = f"/opt/jinhanz/results/bdd/xai_saliency_maps_yolov5s_{sigma}/fullgradcamraw_vehicle"
#     target_img_name = '117'
#     ext = 'jpg'
#     combine_images(root_dir,target_img_name,ext,sigma)

imgs = [
"airplane_167540",
"airplane_338325",
"apple_216277",
"backpack_177065",
"backpack_370478",
"banana_279769",
"banana_290619",
"baseball bat_129945",
"baseball bat_270474",
"baseball glove_162415",
"baseball glove_515982",
"bear_519611",
"bear_521231",
"bed_468245",
"bed_491757",
"bench_310072",
"bench_350607",
"bicycle_203317",
"bicycle_426166",
"bird_100489",
"bird_404568",
"boat_178744",
"boat_442822",
"book_167159",
"bottle_385029",
"bottle_460929",
"bowl_205834",
"bowl_578871",
"broccoli_61658",
"broccoli_389381",
"bus_106048",
"bus_226154",
"cake_119677",
"cake_189451",
"car_227511",
"car_310072",
"carrot_130613",
"carrot_287667",
"cat_101420",
"cat_558073",
"cell phone_396729",
"cell phone_480212",
"chair_81061",
"chair_190236",
"clock_60363",
"couch_29596",
"couch_31735",
"cow_361268",
"cow_545958",
"cup_226171",
"cup_323151",
"dining table_385029",
"dining table_480122",
"dog_331075",
"dog_357459",
"donut_109798",
"donut_148957",
"elephant_83113",
"elephant_97230",
"fire hydrant_293071",
"fire hydrant_344909",
"fork_243626",
"fork_250766",
"frisbee_139872",
"frisbee_357459",
"giraffe_287545",
"giraffe_289659",
"handbag_250127",
"handbag_383842",
"horse_382088",
"horse_439715",
"hot dog_311950",
"hot dog_400082",
"keyboard_66635",
"keyboard_378099",
"kite_24027",
"knife_116206",
"knife_227985",
"laptop_360951",
"laptop_482970",
"microwave_91615",
"microwave_207538",
"motorcycle_462756",
"motorcycle_499622",
"mouse_68765",
"orange_50679",
"orange_386277",
"oven_802",
"oven_446005",
"parking meter_333956",
"parking meter_568147",
"person_278705",
"person_562243",
"pizza_276285",
"pizza_294831",
"potted plant_407614",
"potted plant_473219",
"refrigerator_498463",
"refrigerator_536947",
"remote_430286",
"remote_476810",
"sandwich_417608",
"sandwich_465430",
"scissors_324715",
"scissors_340930",
"sheep_278353",
"sheep_410428",
"sink_51598",
"sink_466085",
"skateboard_71877",
"skateboard_229553",
"skis_30504",
"skis_342397",
"snowboard_393469",
"snowboard_425906",
"spoon_88040",
"spoon_248314",
"sports ball_22935",
"sports ball_60102",
"stop sign_724",
"stop sign_100283",
"suitcase_23023",
"suitcase_350019",
"surfboard_52507",
"surfboard_554595",
"teddy bear_82180",
"teddy bear_205542",
"tennis racket_270908",
"tennis racket_394559",
"tie_133343",
"tie_244496",
"toilet_42276",
"toilet_85576",
"toothbrush_160666",
"traffic light_133087",
"train_90155",
"train_539143",
"truck_295420",
"truck_334006",
"tv_104666",
"tv_453722",
"umbrella_180487",
"umbrella_455157",
"vase_376478",
"vase_521282",
"wine glass_25394",
"wine glass_146489",
"zebra_449406",
"zebra_491613",
]

imgs = [
    "3",
"30",
"33",
"36",
"49",
"54",
"66",
"67",
"68",
"74",
"87",
"100",
"113",
"117",
"126",
"133",
"141",
"171",
"178",
"180",
"183",
"188",
"192",
"209",
"241",
"245",
"269",
"273",
"293",
"327",
"329",
"342",
"344",
"376",
"381",
"388",
"401",
"407",
"431",
"437",
"441",
"447",
"452",
"503",
"514",
"559",
"570",
"585",
"600",
"610",
"611",
"617",
"629",
"668",
"670",
"692",
"698",
"699",
"715",
"740",
"758",
"764",
"803",
"804",
"822",
"833",
"842",
"843",
"852",
"856",
"859",
"867",
"896",
"914",
"930",
"940",
"941",
"942",
"976",
"980",
"987",
"1026",
"1031",
"1043",
"1047",
"1065",
"1090",
"1099",
"1100",
"1109",
"1112",
"1114",
"1145",
"1149",
"1178",
"1183",
"1211",
"1223",
"1226",
"1232",
"1236",
"1252",
"1278",
"1319",
"1331",
"1353",
"1357",
"1365",
]

# imgs = ['chair_81061','elephant_97230']

# imgs = ['585','930','117']

for target_img_name in imgs:
    for is_act in ['xai_saliency','activation']:
        for rescale_method in ['bilinear','optimize_faithfulness']:
            root_dir = f"/opt/jinhanz/results/{rescale_method}/bdd/{is_act}_maps_fasterrcnn/"
            ext = '-res.jpg'
            combine_images(is_act,rescale_method,root_dir,target_img_name,ext)