function dstCands = x_jumpPerform6(srcCands, srcFrIdx, dstFrIdx, param, options, gbvsParam, ofParam, poseletModel, rf, cache)
% Performs jump from source candidates to the destination ones. Destination
% candidates are created using jumpCandidates. Both source and destination
% frame are preprocessed using preprocessFrame, that supports caching.
% Optimized for CVPR13 model and does not create srcFr.
% 
% dstCands = jumpPerform6(srcCands, srcFrIdx, dstFrIdx, param, options, gbvsParam, ofParam, poseletModel, rf, cache)
% 
% INPUT
%   srcCands        cell array of source candidates at the source frame. 
%                   This structure can be created using jumpCandidates. 
%                   The required fields in every cell:
%       .point          the candidate point
%       .score          score of the candidate. Used in weighted sum of the
%                       jump probabilities
%   srcFrIdx        index of source frame in video
%   dstFrIdx        index of destination frame in video
%   param           parameters (of tracking). Must include 'videoReader'
%                   field which is input video stream (as VideoReader)
%   options         options for the call. Used for calling jumpCandidates
%                   and jumpPairwiseFeatures function. See those for the
%                   fields.
%   gbvsParam       GBVS saliency detection parameters. Created by configureDetectors
%   ofParam         optical flow parameters. Created by configureDetectors
%   poseletModel    poselets detection model. Created by configureDetectors
%   rf              trained random forest model (from regRF_train)
%   cache           structure used for caching. See preprocessFrame for
%                   details
% 
% OUTPUT
%   dstCands        created destination candidates, with jump probability
%                   in each. Those are created by jumpCandidates. Score of
%                   each is stored to 'candScore' field and 'score' field
%                   is updated with jump probability

% collect sources
nSrc = length(srcCands);
srcPts = zeros(nSrc, 2);
srcScore = zeros(nSrc, 1);

for ic = 1:nSrc
    srcPts(ic, :) = srcCands{ic}.point;
    srcScore(ic) = srcCands{ic}.score;
end
if (all(srcScore == 0))
    srcScore(:) = 1;
end

% preprocess frames
srcFr = struct('index', srcFrIdx);
dstFr = x_preprocessFrames(param.videoReader, dstFrIdx, gbvsParam, ofParam, poseletModel);

% destination candidates
maps = cat(3, (dstFr.ofx.^2 + dstFr.ofy.^2), dstFr.saliency);
dstCands = x_jumpCandidates(dstFr.faces, dstFr.poselet_hit, maps, options);
nDst = length(dstCands);

% features
features = x_jumpPairwiseFeatures6(srcFr, srcCands, dstFr, dstCands, options); % v6, cached
%TODO strange: nDst ~= size(features, 2)
if (nSrc == 1 && size(features, 2) ~= nDst)
    nDst = min(nDst, size(features, 2));
    dstCands = dstCands(1:nDst);
end
jumpProb = zeros(nSrc, nDst);
lblHard = zeros(nSrc, nDst);

% pairwise jumps
for isrc = 1:nSrc % predict for every source
    for idst = 1:nDst
        if (strcmp(options.rfType, 'reg'))
            lbl = regRF_predict(features(:, nDst*(isrc-1)+idst)', rf); % RF reg
        elseif (strcmp(options.rfType, 'reg-dist'))
            lbl = regRF_predict(features(:, nDst*(isrc-1)+idst)', rf); % RF regression for distance
        elseif (strcmp(options.rfType, 'class'))
            [lblHard(isrc, idst), lbl] = classRF_predict(features(:, nDst*(isrc-1)+idst)', rf); % RF classification
            lbl = lbl(2) / sum(lbl);
        end
        jumpProb(isrc, idst) = lbl;
    end
end

% labels post-processing
if (strcmp(options.rfType, 'reg'))
    % jumpProb(jumpProb < pTh) = 0;
    jumpProb = jumpProb - min(jumpProb(:)); % make all positive
elseif (strcmp(options.rfType, 'reg-dist'))
    pmin = min(jumpProb(:));
    pmax = max(jumpProb(:));
    if (pmin == pmax)
        jumpProb(:) = 0;
    else
        jumpProb = 1 - ((jumpProb - pmin) / (pmax - pmin));
    end
elseif (strcmp(options.rfType, 'class'))
    % use hard labeling
    lblHard(~any(lblHard, 2), :) = 1;
    jumpProb = jumpProb .* lblHard;
end

% score normalization
jps = sum(jumpProb, 2);
idx = find(jps > 0);
if (~isempty(idx))
    jumpProb(idx, :) = jumpProb(idx, :) ./ repmat(jps(idx), [1, nDst]);
end
score = sum(jumpProb .* repmat(srcScore, [1, nDst]), 1);
if (max(score > 0))
    score = score ./ max(score);
end

% update destination scores
for ic = 1:nDst
    dstCands{ic}.candScore = dstCands{ic}.score;
    dstCands{ic}.score = score(ic);
end
