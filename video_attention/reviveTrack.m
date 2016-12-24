function [cands, trackRect, trackRectScore] = reviveTrack(frames, frId, prevMap, nTracks, gbvsParam, ofParam, poseletModel, cache, options, param, useCenter, rf, rectSz, m, n)

[srcPts, srcScore, ~] = sourceCands(prevMap, options);
nSrc = size(srcPts, 1);

srcFr = preprocessFrames(param.videoReader, frames(frId)-1, gbvsParam, ofParam, poseletModel, cache);
dstFr = preprocessFrames(param.videoReader, frames(frId), gbvsParam, ofParam, poseletModel, cache);

maps = cat(3, (dstFr.ofx.^2 + dstFr.ofy.^2), dstFr.saliency);
[dstCands, dstPts, dstScore, dstType] = jumpCandidates(dstFr.faces, dstFr.poselet_hit, maps, options);
% [dstPts, dstScore, dstType] = jumpCands(dstFr.faces, dstFr.poselet_hit, useCenter, maps, options);
nDst = size(dstPts, 1);

% features
[features, ~] = jumpPairwiseFeatures(srcFr, srcPts, dstFr, dstPts, dstType, options);
jumps.src = srcPts';
jumps.dst = dstPts';
jumps.srcW = srcScore(:)';
jumps.p = zeros(nSrc, nDst);

% pairwise jumps
for isrc = 1:nSrc % predict for every source
    for idst = 1:nDst
        lbl = regRF_predict(features(:, nDst*(isrc-1)+idst)', rf); % RF reg
        jumps.p(isrc, idst) = lbl;
    end
end


% rectangles to track
pThresh = 0;
[trackRect, trackRectScore] = candidate2Rect([n, m], jumps, pThresh, rectSz');
[sr, sri] = sort(trackRectScore, 'descend');
cands.rect = trackRect;
cands.score = trackRectScore;
cands.bestRectIdx = sri(1:nTracks);
trackRect = trackRect(:, sri(1:nTracks));
trackRectScore = trackRectScore(:, sri(1:nTracks));
