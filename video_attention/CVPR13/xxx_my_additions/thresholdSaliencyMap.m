function bmap = thresholdSaliencyMap(sMap, p)

ssv = sort(sMap(:), 'descend');
bmap = (sMap >= ssv(ceil(p * length(ssv))));

