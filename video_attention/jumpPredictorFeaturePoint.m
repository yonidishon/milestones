function f = jumpPredictorFeaturePoint(srcFr, srcCands, dstFr, dstCands, fnum, options)
%

nbins = 256;
ndbins = 4;
f = zeros(fnum, 1);
curi = 1;

% color
s = rgb2ycbcr(double(srcFr.image));
d = rgb2ycbcr(double(dstFr.image));
for i = 1:3
    f(curi+i-1) = compareHist(s(:,:,i), d(:,:,i), nbins);
end
curi = curi + 3;
clear s d;

% motion
s = srcFr.ofx.^2 + srcFr.ofy.^2;
d = dstFr.ofx.^2 + dstFr.ofy.^2;
f(curi) = compareHist(s, d, nbins);
curi = curi + 1;
clear s d;

% number of candidates
ns = length(srcCands);
nd = length(dstCands);
f(curi) = ns;
f(curi+1) = nd;
curi = curi + 2;

% distances between candidates
srcPts = zeros(ns, 2);
dstPts = zeros(nd, 2);
for i = 1:ns, srcPts(i, :) = srcCands{i}.point; end;
for i = 1:nd, dstPts(i, :) = dstCands{i}.point; end;

D = pdist2(srcPts, dstPts, 'euclidean');
h = hist(min(D), ndbins);
f(curi:curi+ndbins-1) = h(:);

% remove nans
f(isnan(f)) = 0;

function d = compareHist(d1, d2, nbins)
p1 = hist(d1(:), nbins);
p2 = hist(d2(:), nbins);

d = sum((p1 - p2) .^ 2 ./ (p1 + p2)) ./ 2;
