function output_map = gen_heatmap_single(Sub, opt)

d1 = size(Sub,1);

summary = zeros(d1,4);
summary(:,1) = 1;
summary(:,2) = Sub(:,3);
summary(:,3) = Sub(:,1);
summary(:,4) = Sub(:,2);

save('data1.mat','summary');

cfg=[];
cfg.xSize = opt.xSize; %height of the image, in pixel
cfg.ySize = opt.ySize; %wideth of the image, in pixel
cfg.columnx=3;
cfg.columny=4;
cfg.columnduration=2;
cfg.columnitem=1;
cfg.dataset1= 1;
cfg.maptype = 1;
% cfg.backgroundfile=[];    %'11.tiff'
cfg.smoothingpic = opt.gaussian_smooth_val;
output_map = imap_simple(cfg);

end