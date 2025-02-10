import os

original_annotation_path = {
    'vehicle': '/home/jinhanz/cs/data/bdd/orib_veh_id_task0922_label',
    'human': '/home/jinhanz/cs/data/bdd/orib_hum_id_task1009_label'
}

mapped_annotation_path = {
    'vehicle': '/home/jinhanz/cs/data/bdd/orib_veh_id_task0922_mscoco_label',
    'human': '/home/jinhanz/cs/data/bdd/orib_hum_id_task1009_mscoco_label'
}

category_mapping = {
    '0': '0', # person
    '1': '0', # ‘rider’
    '2': '2', # ‘car’
    '3': '5', # ‘bus’
    '4': '7' # ‘truck’
}

for dataset in ['vehicle', 'human']:
    os.makedirs(mapped_annotation_path[dataset], exist_ok=True)

    for filename in os.listdir(original_annotation_path[dataset]):

        if 'txt' not in filename:
            continue

        with open(os.path.join(original_annotation_path[dataset], filename), 'r') as f:
            lines = f.readlines()

        with open(os.path.join(mapped_annotation_path[dataset], filename), 'w') as f:
            for line in lines:
                new_line = category_mapping[line[0]] + line[1:]
                f.write(new_line)