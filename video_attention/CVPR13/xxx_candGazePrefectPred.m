function [perfect_predmap] = xxx_candGazePrefectPred(gazePts, cands,h,w)
%

gazePts = gazePts(~isnan(gazePts(:, 1)), :);
nc = length(cands);
candPts = zeros(nc, 2);
candRad = zeros(nc, 1);
for ic = 1:nc
    candPts(ic, :) = cands{ic}.point(:)';
    candRad(ic) = mean(sqrt(diag(cands{ic}.cov)));
end
% D is an mxn where length(gazePts)==m and length(candPts)==n distances
D = pdist2(gazePts, candPts,'euclidean');
% Now I need to count the numbers of rows in D for each column that are
% smaller than the candRad
smlthan = bsxfun(@ge,candRad',D);
if size(smlthan,1) > 1
    cand_weight = sum(smlthan);
elseif any(smlthan)
    [~,idx]=max(smlthan);
    cand_weight = smlthan(idx)==smlthan;
else
    cand_weight = ones(1,length(candRad));
end
perfect_predmap = points2GaussMap(candPts',cand_weight,0,[w,h],candRad);