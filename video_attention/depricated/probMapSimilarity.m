function [simil, det] = probMapSimilarity(pmap1, pmap2, simMeas, varargin)

narg = length(varargin);
det = [];

if (strcmp(simMeas, 'chisq')) % chi-square measure
    p1 = pmap1(:) / sum(pmap1(:));
    p2 = pmap2(:) / sum(pmap2(:));
    simil = sum((p1 - p2) .^ 2 ./ (p1 + p2)) ./ 2;
    
elseif (strcmp(simMeas, 'auc')) % area under ROC curve
    if (isempty(pmap1))
        simil = nan;
        return;
    end
    
    if (narg == 1)
        rocStops = varargin{1};
    else
        rocStops = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
    end
    
    % pmap1 are points
    [m, n] = size(pmap2);
    validIdx = (pmap1(:,2) > 0 & pmap1(:,2) <= m & pmap1(:,1) > 0 & pmap1(:,1) <= n);
    dt = ceil(pmap1(validIdx, :));
    npt = length(rocStops);
    roc = zeros(size(rocStops));
    auc = 0;
    
    ind = sub2ind(size(pmap2), dt(:, 2), dt(:, 1));
    
    for ipt = 1:npt % for every point
        bmap = thresholdSaliencyMap(pmap2, rocStops(ipt));
        bhit = bmap(ind);
        roc(ipt) = sum(bhit) / length(bhit);
        
        if (ipt == 1)
            auc = auc + roc(ipt)/2 * rocStops(ipt);
        else
            auc = auc + (roc(ipt) + roc(ipt-1))/2 * (rocStops(ipt)-rocStops(ipt-1));
        end
    end
    
    det = roc;
    simil = auc;
elseif(strcmp(simMeas, 'ishit')) % checks if there hits to all regions map
    if (isempty(pmap1))
        simil = nan;
        return;
    end
    
    if (narg == 1)
        th = varargin{1};
    else
        th = 0.1; % top 10%
    end
    
    % pmap1 are points
    [m, n] = size(pmap2);
    validIdx = (pmap1(:,2) > 0 & pmap1(:,2) <= m & pmap1(:,1) > 0 & pmap1(:,1) <= n);
    dt = ceil(pmap1(validIdx, :));
    ind = sub2ind(size(pmap2), dt(:, 2), dt(:, 1));
    
    bmap = thresholdSaliencyMap(pmap2, th);
    [lblMap, nlbl] = bwlabel(bmap);
    bhit = lblMap(ind);
    uh = unique(bhit);
    if (uh(1) == 0)
        uh = uh(2:end);
    end
    simil = length(uh) / nlbl;
else
    error('Unsupported similarity measure: %s', simMeas)
end

function bmap = thresholdSaliencyMap(pmap, p)

ssv = sort(pmap(:), 'descend');
th = ssv(ceil(p * length(ssv)));
bmap = (pmap >= th);
