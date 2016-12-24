function mp = candidate2map(cands, sz, scale)
% Converts candidates to dense heat-map
% 
% mp = candidate2map(cands, sz)
%
% INPUT
%   cands   cell array of candidates in the frame. Each cell includes
%       .point      a 2D point in [x y] format
%       .candCov    covariance of the Gaussian around each point
%       .score      the score of the candidate, used as attention
%                   probability
%   sz      frames size, [w h]
%   scale   [1] scaling factor of the Gaussian blob, optional
%
% OUTPUT
%   mp      grayscale heat-map h X w, [0...1]

if (~exist('scale', 'var'))
    scale = 1;
end

mp = zeros(sz(2), sz(1));
nc = length(cands);
[X, Y] = meshgrid(1:sz(1), 1:sz(2));

for ic = 1:nc
    mu = cands{ic}.point;
    s = inv(scale * cands{ic}.candCov);
    w = cands{ic}.score;
    
    fg = w * exp(-1/2 .* (s(1,1).*(X-mu(1)).^2 + s(2,2).*(Y-mu(2)).^2 + (s(2,1)+s(1,2)).*(X-mu(1)).*(Y-mu(2))));
%     fg = exp(-((X - pts(1,ipt)).^2/2/sigma(ipt)^2 + (Y - pts(2,ipt)).^2/2/sigma(ipt)^2));
%     fg = w(ipt) .* fg ./ max(fg(:));
    
    mp = mp + fg;
end

if (max(mp(:) > 0))
    mp = mp ./ max(mp(:));
end
