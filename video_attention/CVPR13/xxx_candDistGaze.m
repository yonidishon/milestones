function [dst, hit, cand_hit_ratio,hit_type] = xxx_candDistGaze(gazePts, cands)
%

gazePts = gazePts(~isnan(gazePts(:, 1)), :);
nc = length(cands);
candPts = zeros(nc, 2);
candRad = zeros(nc, 1);
for ic = 1:nc
    candPts(ic, :) = cands{ic}.point(:)';
    candRad(ic) = mean(sqrt(diag(cands{ic}.cov)));
end

D = pdist2(gazePts, candPts);
[dm, ii] = min(D, [], 2);
dm = sqrt(dm);
dst = mean(dm(~isnan(dm)));
if isempty(dm) && isempty(candRad(ii))
    hit = NaN;
    cand_hit_ratio = NaN;
    hit_type =[];
    return;
end
hit = double(dm <= 2 * candRad(ii));
Ds = sqrt(D);
candhit = double(Ds <= repmat(2*candRad',size(Ds,1),1));
cand_hit_ratio = sum((sum(candhit) > 1))/size(candhit,2);
hit_type = cellfun(@(x)extractfield(x,'type'),cands(find((sum(candhit) > 1))));
