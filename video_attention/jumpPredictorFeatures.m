function [features, distances, labels] = jumpPredictorFeatures(srcFr, srcCands, dstFr, dstCands, options, cache)
% creates jump predictor features
%
% [features, distances, labels] = jumpPairwiseFeatures(srcFr, srcCands, dstFr, dstCands, options, cache)
%
% INPUT
%   srcFr, dstFr    source / destination frames, see preprocessFrames for
%                   details
%   srcCands, dstCands  source / destination points, cell array
%   options         options, used fields
%       .useLabel       if true labels will be creates (+1 if agree with
%                       ground truth, -1 if not
%       .dstGroundTruth ground truth map in destination frame. Used only if
%                       'useLabel' is true
%       .srcGroundTruth ground truth map in source frame. Used only if
%                       'useLabel' is true
%       .dstGroundTruthPts  ground truth points in destination frame. Used
%                           only if 'useLabel' is true
%       .gazeThreshold  allowed deviation from true point
%       .distType       distance type to use. 'mahal' for Mahalanobis,
%                       'euc' for Euclidian
%       .featureNormMean    
%       .featureNormSdt     
%       .featureIdx     indices of features to use
% OUTPUT
%   features    
%   distances   
%   labels      

fvMax = 10;
if (~isfield(options, 'featureIdx'))
    options.featureIdx = 1:fvMax;
end

if (exist('cache', 'var'))
    cacheD = fullfileCreate(cache.featureRoot, dstFr.videoName);
    cacheFile = fullfile(cacheD, sprintf('%d__%d.mat', srcFr.index, dstFr.index));
else
    cacheFile = [];
end

distances = inf;
labels = 0;
% distances = zeros(1, nSrc * nDst);

if (~isempty(cacheFile) && ~cache.renewFeatures && exist(cacheFile, 'file')) % load from cache
    s = load(cacheFile);
    features = s.features;
    distances = s.distances;
    labels = s.labels;
    clear s;
else % calculate features
    % find destination ground truth points
    if (isfield(options, 'useLabel') && options.useLabel)
        if (~isfield(options, 'distType') || strcmp(options.distType, 'mahal')) % mahalanobis distance
            error('No distance defined');
%             options.distType = 'mahal';
%             dstPts = zeros(nDst, 2);
%             for idst = 1:nDst
%                 dstPts(idst, :) = dstCands{idst}.point;
%             end
%             if (isfield(options, 'dstGroundTruthPts') && size(options.dstGroundTruthPts, 1) > 2)
%                 mahalD = mahal(dstPts, options.dstGroundTruthPts);
%             else
%                 mahalD = inf(nDst, 1);
%             end
            
        elseif(strcmp(options.distType, 'euc'))
            % euclidian distance
            gazeTh = options.gazeThreshold * dstFr.height;
            rects = denseMap2Rects(options.dstGroundTruth, options.posPer, 0);
            if (~isempty(rects))
                dstGtPts = rects(:, [1, 2]) + rects(:, [3, 4]) / 2;
            else
                dstGtPts = [];
            end
            rects = denseMap2Rects(options.srcGroundTruth, options.posPer, 0);
            if (~isempty(rects))
                srcGtPts = rects(:, [1, 2]) + rects(:, [3, 4]) / 2;
            else
                srcGtPts = [];
            end
        end
    end
    
    % calculate features
    features = jumpPredictorFeaturePoint(srcFr, srcCands, dstFr, dstCands, fvMax, options);
    
    % set label
    if (isfield(options, 'useLabel') && options.useLabel)
        if (isempty(dstGtPts) || isempty(srcGtPts)) % no jump
            labels = -1;
        else
            D = pdist2(srcGtPts, dstGtPts, 'euclidean');
            distances = mean(min(D));
            if (distances <= gazeTh)
                labels = 1;
            else
                labels = -1;
            end
        end
    end
    
    % save
    if (~isempty(cacheFile))
        save(cacheFile, 'features', 'distances', 'labels');
    end
end

% use only needed features
features = features(options.featureIdx, :);

% normalize features
if (isfield(options, 'featureNormMean'))
    features = features - options.featureNormMean(:);
    features = features ./ options.featureNormStd(:);
end
