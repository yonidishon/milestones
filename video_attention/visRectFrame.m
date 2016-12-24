function [imrect, imo] = visRectFrame(fr, rect, score, cmap)
% Overlays scored rectangles on frame
% 
% [imrect, imo] = visRectFrame(fr, rect, score, cmap)
% [imrect, imo] = visRectFrame(fr, rect, score)
%
% INPUT
%   fr      color frame to use. Will be comverted to grayscale for better
%           visualization
%   rect    matrix of destination rectangles, 4 X m. Each column is 
%           [x y w h]'
%   score   score of every rectangle, [0...1]
%   cmap    [] colormap to use in scores, if empty all the rectangles are
%           red

if (~exist('cmap', 'var'))
    cmap = [1 0 0];
end

m = size(rect, 2);
[hFr, wFr, ~] = size(fr);

if (~exist('score', 'var'))
    score = ones(1, m);
end
score = score ./ 2 + 0.5; % scale to [0.5 ... 1] for visibility
score(isnan(score)) = 0; % remove NaNs
% cmIdx = score * (size(cmap, 1) - 1) + 1;

% adjust frame
gim = rgb2gray(fr);
imrect = zeros(size(gim));
gim = imadjust(gim, [0; 1], [0.3 0.7]);
imo = repmat(gim, [1 1 3]);

% add rectangles
for i = 1:m
    if (all(rect(:,i) == 0)), continue; end;
%     cl = cmap(cmIdx(i), :);
    x = rect(1,i); y = rect(2,i); w = rect(3,i); h = rect(4,i);
%     hb = repmat(reshape(cl, [1 1 3]), [1 w+1 1]);
%     vb = repmat(reshape(cl, [1 1 3]), [h+1 1 1]);
    
%     imo(max(y,1), max(x,1):min(x+w,wFr), :) = hb;
%     imo(min(y+h,hFr), max(x,1):min(x+w,wFr), :) = hb;
%     imo(max(y,1):min(y+h,hFr), max(x,1), :) = vb;
%     imo(max(y,1):min(y+h,hFr), min(x+w,wFr), :) = vb;
    
    imrect(max(y,1), max(x,1):min(x+w,wFr)) = score(i);
    imrect(min(y+h,hFr), max(x,1):min(x+w,wFr)) = score(i);
    imrect(max(y,1):min(y+h,hFr), max(x,1)) = score(i);
    imrect(max(y,1):min(y+h,hFr), min(x+w,wFr)) = score(i);
end
