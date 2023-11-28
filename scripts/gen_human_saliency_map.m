clear;
Path.RawDataPath = 'H:\OneDrive - The University Of Hong Kong\bdd\results\explanation\231018_vehicle_whole_screen_vb_fixed_pos\exp_fixations_split_by_img';
Path.CodePath = pwd;
addpath(genpath(Path.CodePath));
Path.picSavePath = fullfile('H:\OneDrive - The University Of Hong Kong\bdd\results\explanation\231018_vehicle_whole_screen_vb_fixed_pos\',['human_saliency_map']);
CheckIfDirExist(Path);

fileDir = dir(fullfile(Path.RawDataPath, ['*.xlsx']));

for i = 1:numel(fileDir)

    curDir = fullfile(fileDir(i).folder, fileDir(i).name);
    
    intA = xlsread(curDir);
    
    opt.gaussian_smooth_val = 30;   % 30 pixels, correspond to 1 degree (visual angle)
    [output_map] = gen_heatmap_single(intA(:,1:3), opt);
    output_map_norm = (output_map-min(output_map,[],'all'))./(max(output_map,[],'all')-min(output_map,[],'all'));
    
    save(fullfile(Path.picSavePath, [fileDir(i).name(1:end-5) '_GSmo_' num2str(opt.gaussian_smooth_val) '.mat']), 'output_map_norm');

end


%% Support Functions
function output_map = gen_heatmap_single(Sub, opt)

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
% cfg.backgroundfile=[];    %'11.tiff'
cfg.smoothingpic = opt.gaussian_smooth_val;
output_map = imap_gy(cfg);

end



