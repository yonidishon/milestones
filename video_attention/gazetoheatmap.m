function [heatmap]=gazetoheatmap(fr,gazepts,sigma)
cmap = jet(256);
ncol = size(cmap, 1);
p=points2GaussMap(gazepts',ones(1, size(gazepts, 1)), 0, [size(fr,2),size(fr,1)], sigma);
p=p./max(p(:));
alphaMap = 0.7 * repmat(p, [1 1 3]);
rgbHM = ind2rgb(round(p * ncol), cmap);
gim = rgb2gray(fr);
gim = imadjust(gim, [0; 1], [0.3 0.7]);
gf = repmat(gim, [1 1 3]);
heatmap= rgbHM .* alphaMap + im2double(gf) .* (1-alphaMap);
end