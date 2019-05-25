clc
clear
%% KITTI2012
img_list = dir('./KITTI2012/original/colored_0/*.png');
for idx_file = 1:2:length(img_list)
    file_name = img_list(idx_file).name;
    idx_name = find(file_name == '_');
    file_name = file_name(1:idx_name-1);
    
    img_0 = imread(['./KITTI2012/original/colored_0/',img_list(idx_file).name]);
    img_1 = imread(['./KITTI2012/original/colored_1/',img_list(idx_file).name]);
    
    %% x4
    scale_list = [4];
    for idx_scale = 1:length(scale_list)
        scale = scale_list(idx_scale);
        
        %% generate HR & LR images
        img_hr_0 = modcrop(img_0, scale);
        img_hr_1 = modcrop(img_1, scale);
        img_lr_0 = imresize(img_hr_0, 1/scale, 'bicubic');
        img_lr_1 = imresize(img_hr_1, 1/scale, 'bicubic');
        
        mkdir(['./KITTI2012/hr']);
        mkdir(['./KITTI2012/hr/', file_name]);
        mkdir(['./KITTI2012/lr_x', num2str(scale)]);
        mkdir(['./KITTI2012/lr_x', num2str(scale), '/', file_name]);
        
        imwrite(img_hr_0, ['./KITTI2012/hr/', file_name, '/hr0.png']);
        imwrite(img_hr_1, ['./KITTI2012/hr/', file_name, '/hr1.png']);
        imwrite(img_lr_0, ['./KITTI2012/lr_x', num2str(scale), '/', file_name, '/lr0.png']);
        imwrite(img_lr_1, ['./KITTI2012/lr_x', num2str(scale), '/', file_name, '/lr1.png']);
    end
end