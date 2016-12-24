function frOut = visCandidates(fr, cands, cmap)

if (~exist('cmap', 'var'))
    cmap = [0.5 0.5 0.5; lines(10)];
end

nc = length(cands);
[m, n, ~] = size(fr);
frg = im2double(repmat(rgb2gray(fr), [1 1 3]));
fri = zeros(m, n);
alpha = 0.5;
rad = 2;

frTr = frg; % visualize tracking rectangles

for ic = 1:nc % for each candidate
    % candidates
    msk = maskEllipse(n, m, cands{ic}.point, cands{ic}.cov, rad)';
    msk = bwperim(msk, 8);
    mskd = double(msk);
    fri = fri .* (1 - mskd) + (cands{ic}.type+1) .* mskd;
    
    % tracking rectangles
    col = ceil(255 * cmap(cands{ic}.type+1, :));
    frTr = bbApply('embed', frTr, cands{ic}.trackRect, 'col', col, 'lw', 2);
end

frOut = [alpha .* ind2rgb(fri, cmap) + (1-alpha) .* frg;
    frTr];
