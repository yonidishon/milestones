function [dst, hit] = candDistGaze(gazePts, cands)
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

hit = double(dm <= 2 * candRad(ii));
