function p = points2GaussMap(pts, w, wTh, sz, sigma, useMax)
% p = points2GaussMap(pts, w, wTh, sz, sigma, useMax)
% p = points2GaussMap(pts, w, wTh, sz, sigma)
%
% Converts points of data to continious heat map by placing a Gaussian at
% every point. The output map is scaled between 0 and 1.
%
% INPUT
%   pts     points to use, 2 X npt matrix. Each point is in a column.
%   w       weights of the points, vector of npt length
%   wTh     threshold below which the points will not be considered
%   sz      size of output map, [width, height]
%   sigma   sigma of Gaussian at each point. If scalar the same Gaussian
%           is be used
%   useMax  [false] if true the maximum of Gaussians is taken instead of 
%           sum    
%
% OUTPUT
%   p       output heat map, scaled to [0...1]

if (~exist('useMax', 'var'))
    useMax = false;
end

m = sz(2);
n = sz(1);
[X, Y] = meshgrid(1:n, 1:m);
p = zeros(m, n);

npt = size(pts, 2);
if (isscalar(sigma))
    sigma = sigma * ones(1, npt);
end

for ipt = 1:npt
    if (w(ipt) < wTh), continue; end
    
    fg = exp(-((X - pts(1,ipt)).^2/2/sigma(ipt)^2 + (Y - pts(2,ipt)).^2/2/sigma(ipt)^2));
    fg = w(ipt) .* fg ./ max(fg(:));
    
    if (useMax)
        p = max(p, fg);
    else
        if isnan(fg),continue;
        else p = p + fg;
        end
        
    end
end

if (max(p(:) > 0))
    p = p ./ max(p(:));
end
