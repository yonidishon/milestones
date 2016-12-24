function [pSmp, nSmp, pReg, nReg] = sampleMapTopBottom(pmap, n, posPer, negPer, mrg)
% Samples randomly n points from top and bottom part of a probability map
%
% [pSmp, nSmp, pReg, nReg] = sampleMapTopBottom(pmap, n, posPer, negPer, mrg)
%
% INPUT
%   pmap            probability map to sample, eg. gaze, saliency, ...
%   n               number of positive and negative samples
%   posPer, negPer  positive and negative persentation of probability to
%                   use
%   mrg             [0] margin from the boundary of the map
%
% OUTPUT
%   pSmp, nSmp      n X 2 arrays of positive / negative sample points in
%                   format [x,y]
%   pReg, nReg      regions from where the positive / negative samples are
%                   taken

if (~exist('mrg', 'var'))
    mrg = 0;
end

sz = size(pmap);

cReg = false(sz);
cReg(mrg+1:sz(1)-mrg, mrg+1:sz(2)-mrg) = true;

ssv = sort(pmap(:), 'descend');
th = ssv(ceil(posPer * length(ssv)));
pReg = (pmap >= th);
pReg = pReg & cReg;

th = ssv(ceil((1-negPer) * length(ssv)));
nReg = (pmap <= th);
nReg = nReg & cReg;

ind = find(pReg);
rind = ceil(rand(n, 1) * length(ind));
[pSmp(:,2), pSmp(:,1)] = ind2sub(sz, ind(rind));

ind = find(nReg);
rind = ceil(rand(n, 1) * length(ind));
[nSmp(:,2), nSmp(:,1)] = ind2sub(sz, ind(rind));

