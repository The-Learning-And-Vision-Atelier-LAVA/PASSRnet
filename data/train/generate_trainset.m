clc
clear
%%
img_list = dir('./Flickr1024/*.png');
for idx_file = 1:2:length(img_list)
    file_name = img_list(idx_file).name;
    idx_name = find(file_name == '_');
    file_name = file_name(1:idx_name-1);
    
    img_0 = imread(['./Flickr1024/',img_list(idx_file).name]);
    img_1 = imread(['./Flickr1024/',img_list(idx_file+1).name]);
    
    %% x2 or x4
    scale_lisht = [4];
    for idx_scale = 1:length(scale_lisht)
        scale = scale_lisht(idx_scale);
        
        %% generate HR & LR images
        img_hr_0 = modcrop(img_0, scale);
        img_hr_1 = modcrop(img_1, scale);
        img_lr_0 = imresize(img_hr_0, 1/scale, 'bicubic');
        img_lr_1 = imresize(img_hr_1, 1/scale, 'bicubic');
        
        %% extract patches of size 30*90 with stride 20
        idx_patch=1;
        for x_lr = 3:20:size(img_lr_0,1)-33
            for y_lr = 3:20:size(img_lr_0,2)-93
                x_hr = (x_lr-1) * scale + 1;
                y_hr = (y_lr-1) * scale + 1;
                hr_patch_0 = img_hr_0(x_hr:(x_lr+29)*scale,y_hr:(y_lr+89)*scale,:);
                hr_patch_1 = img_hr_1(x_hr:(x_lr+29)*scale,y_hr:(y_lr+89)*scale,:);
                lr_patch_0 = img_lr_0(x_lr:x_lr+29,y_lr:y_lr+89,:);
                lr_patch_1 = img_lr_1(x_lr:x_lr+29,y_lr:y_lr+89,:);
                
                mkdir(['./Flickr1024_patches/patches_x', num2str(scale), '/', file_name,'_', num2str(idx_patch, '%03d')]);
                imwrite(hr_patch_0, ['./Flickr1024_patches/patches_x', num2str(scale), '/', file_name, '_', num2str(idx_patch, '%03d'), '/hr0.png']);
                imwrite(hr_patch_1, ['./Flickr1024_patches/patches_x', num2str(scale), '/', file_name, '_', num2str(idx_patch, '%03d'), '/hr1.png']);
                imwrite(lr_patch_0, ['./Flickr1024_patches/patches_x', num2str(scale), '/', file_name, '_', num2str(idx_patch, '%03d'), '/lr0.png']);
                imwrite(lr_patch_1, ['./Flickr1024_patches/patches_x', num2str(scale), '/', file_name, '_', num2str(idx_patch, '%03d'), '/lr1.png']);
                
                idx_patch = idx_patch + 1;
            end
        end
    end
end