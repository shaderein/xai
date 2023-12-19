% Generate human attention maps based for MSCOCO images, which have
% inconsistent image size, whereas human fixations cover the whole screen
%   - Jinhan Zhang 23-12-18

clear;

% DET Hum
% Path.RawDataPath = 'H:\Projects\HKU_XAI_Project\XAI_Similarity_1\Hum_IdTask_Split_Fixation';
% Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\231206 Hum DET\whole_image';
% Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\231206 Hum DET\whole_image_visualize';
% Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\bdd\images\orib_hum_id_task_resized\';

% EXP Hum
Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\bdd\results\explanation\231018_vehicle_whole_screen_vb_fixed_pos\exp_fixations_split_by_img';
Path.matSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\231219 Veh EXP\attention_maps';
Path.picSavePath = 'H:\OneDrive - The University Of Hong Kong\bdd\attention_maps\231219 Veh EXP\visualize';
Path.backgroundPath = 'H:\OneDrive - The University Of Hong Kong\bdd\images\orib_veh_id_task_resized\';


Path.CodePath = pwd;
addpath(genpath(Path.CodePath));

CheckIfDirExist(Path);

fileDir = dir(fullfile(Path.RawDataPath, ['*.xlsx']));

for i = 1:numel(fileDir)

    curDir = fullfile(fileDir(i).folder, fileDir(i).name);
    
    intA = xlsread(curDir);

    img_path = [Path.backgroundPath strrep(fileDir(i).name, 'xlsx', 'jpg')]
    
    opt.gaussian_smooth_val = 30;   
    [output_map] = gen_heatmap_single(intA(:,1:3), img_path, opt);
    output_map_norm = (output_map-min(output_map,[],'all'))./(max(output_map,[],'all')-min(output_map,[],'all'));
    
    % save mat file
    save(fullfile(Path.matSavePath, [fileDir(i).name(1:end-5) '_GSmo_' num2str(opt.gaussian_smooth_val) '.mat']), 'output_map_norm');

    % save plots
    copyfile('dataset1picedge.tiff', fullfile(Path.picSavePath, [fileDir(i).name(1:end-5) '_GSmo_' num2str(opt.gaussian_smooth_val) '.tiff']))

end


%% Support Functions
function output_map = gen_heatmap_single(Sub, img_path, opt)

d1 = size(Sub,1);

summary = zeros(d1,4);
summary(:,1) = 1;
summary(:,2) = Sub(:,3);
summary(:,3) = Sub(:,1);
summary(:,4) = Sub(:,2);

save('data1.mat','summary');

cfg=[];
cfg.xSize= 576; %height of the image, in pixel
cfg.ySize=1024; %wideth of the image, in pixel
cfg.columnx=3;
cfg.columny=4;
cfg.columnduration=2;
cfg.columnitem=1;
cfg.dataset1= 1;
cfg.maptype = 1;
cfg.backgroundfile=img_path;    %'11.tiff'
cfg.smoothingpic = opt.gaussian_smooth_val;
output_map = imap_gy(cfg);

end



