function img_cropped = modcrop(img, scale_factor)
h = size(img, 1);
w = size(img, 2);

img_cropped = img(1:floor(h/scale_factor)*scale_factor, 1:floor(w/scale_factor)*scale_factor, :);
end

