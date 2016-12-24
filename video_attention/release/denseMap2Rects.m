function rects = denseMap2Rects(mp, th, rectSzTh)
% converts a probability map to rectangles
% 
% rects = denseMap2Rects(mp, th, rectSzTh)
% rects = denseMap2Rects(mp, th)
%
% INPUT
%   mp          probability map to convert
%   th          percentage (0...1) of the top value to use
%   rectSzTh    [0] rectangles below this size will be rejected
%
% OUTPUT
%   rects       created rectangles, [nrect X 4]

if (~exist('rectSzTh', 'var'))
    rectSzTh = 0;
end

% return empty rects for flat map
if (isempty(mp) || all(isnan(mp(:))) || (min(mp(:)) == max(mp(:))))
   rects = [];
   return;
end

bmap = thresholdSaliencyMap(mp, th);
[lbl, nlbl] = bwlabel(bmap);

rects = zeros(nlbl, 4);
for i = 1:nlbl
    [row, col] = find(lbl == i);
    x = min(col);
    y = min(row);
    w = max(col) - x;
    h = max(row) - y;
    rects(i, :) = [x, y, w, h];
end

rects = rects((rects(:,3) > rectSzTh & rects(:, 4) > rectSzTh), :);
