function [frg] = xxx_candsvis(cands,visframe)
% This function visualize the candidates over the visual frame
alpha = 0.5;
rad = 2;
ncol = 256;
frg = im2double(repmat(rgb2gray(visframe), [1 1 3]));
candMap = zeros(size(frg,1), size(frg,2));
for ic = 1:length(cands) % for each candidate
    msk = maskEllipse(size(frg,2), size(frg,1), cands{ic}.point, cands{ic}.cov, rad)';
    msk = bwperim(msk, 8);
    mskd = double(msk);
    candMap = candMap .* (1 - mskd) + (floor((cands{ic}.type-1)/6 * ncol/2 + ncol/2)) .* mskd;
end
frg = frg + repmat(candMap,[1,1,3]);
