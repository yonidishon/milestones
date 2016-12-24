function p = rectDet2GaussMap(rectDet, scoreDet, sz, sc)

m = sz(2);
n = sz(1);
[X, Y] = meshgrid(1:n, 1:m);
p = zeros(m, n);

if (~isempty(rectDet))
    nf = size(rectDet, 1);
    
    for i = 1:nf
        xf = rectDet(i,1) + rectDet(i,3)/2;
        yf = rectDet(i,2) + rectDet(i,4)/2;
        fg = exp(-((X - xf).^2/2/(sc*rectDet(i,3))^2 + (Y - yf).^2/2/(sc*rectDet(i,4))^2));
        fg = scoreDet(i) .* fg ./ max(fg(:));
        
        p = max(p, fg);
    end
end
