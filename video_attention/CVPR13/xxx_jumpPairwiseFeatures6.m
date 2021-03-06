function [features, distances, labels, jumps] = xxx_jumpPairwiseFeatures6(srcFr, srcCands, dstFr, dstCands, options, cache)
% creates pairwise jump features
% Optimized for CVPR13 model and does not use srcFr. It can be set to empty
% to save preprocessing
%
% [features, labels, jumps] = jumpPairwiseFeatures(srcFr, srcCands, dstFr, dstCands, options)
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
%   jumps       array of jumps, [nsrc * ndst X 6], where each row is a
%               jump: [src_x, src_y, dst_x, dst_y, dst_type, label]

fvMax = 22;

if (exist('cache', 'var'))
    cacheD = fullfileCreate(cache.featureRoot, dstFr.videoName);
    cacheFile = fullfile(cacheD, sprintf('%d__%d.mat', srcFr.index, dstFr.index));
else
    cacheFile = [];
end

nSrc = length(srcCands);
nDst = length(dstCands);
features = zeros(fvMax, nSrc * nDst);
labels = zeros(1, nSrc * nDst);
distances = zeros(1, nSrc * nDst);
jumps = zeros(nSrc * nDst, 6);

if (~isempty(cacheFile) && ~cache.renewFeatures && exist(cacheFile, 'file')) % load from cache
    s = load(cacheFile);
    features = s.features;
    distances = s.distances;
    labels = s.labels;
    jumps = s.jumps;
    clear s;
else % calculate features
    % find destination ground truth points
    if (isfield(options, 'useLabel') && options.useLabel)
        if (~isfield(options, 'distType') || strcmp(options.distType, 'mahal')) % mahalanobis distance
            options.distType = 'mahal';
            dstPts = zeros(nDst, 2);
            for idst = 1:nDst
                dstPts(idst, :) = dstCands{idst}.point;
            end
            if (isfield(options, 'dstGroundTruthPts') && size(options.dstGroundTruthPts, 1) > 2)
                mahalD = mahal(dstPts, options.dstGroundTruthPts);
            else
                mahalD = inf(nDst, 1);
            end
            
        elseif(strcmp(options.distType, 'euc'))
            % euclidian distance
            rects = denseMap2Rects(options.dstGroundTruth, options.posPer, 0);
            gazeTh = options.gazeThreshold * dstFr.height;
            if (~isempty(rects))
                gtPts = rects(:, [1, 2]) + rects(:, [3, 4]) / 2;
            else
                gtPts = [];
            end
        end
    end
    
    for isrc = 1:nSrc
        for idst = 1:nDst
            features(:, nDst*(isrc-1)+idst) = xxx_jumpFeaturePoint6(srcFr, srcCands{isrc}, dstFr, dstCands{idst}, options);
            
            % set label
            if (isfield(options, 'useLabel') && options.useLabel)
                if (dstCands{idst}.type == 6) % destination is gaze
                    labels(nDst*(isrc-1)+idst) = 1;
                else
                    if (strcmp(options.distType, 'mahal')) % mahalanobis distance
                        distances(nDst*(isrc-1)+idst) = mahalD(idst);
                        if (mahalD(idst) <= options.gazeThreshold)
                            labels(nDst*(isrc-1)+idst) = 1;
                        else
                            labels(nDst*(isrc-1)+idst) = -1;
                        end
                    elseif strcmp(options.distType, 'euc') % euclidian distance
                        if (isempty(gtPts))
                            labels(nDst*(isrc-1)+idst) = -1;
                            distances(nDst*(isrc-1)+idst) = inf;
                        else
                            D = pdist2(gtPts, dstCands{idst}.point, 'euclidean');
                            distances(nDst*(isrc-1)+idst) = min(D);
                            if (min(D) <= gazeTh)
                                labels(nDst*(isrc-1)+idst) = 1;
                            else
                                labels(nDst*(isrc-1)+idst) = -1;
                            end
                        end
                    end
                end
            end
            
            % prepare jumps
            jumps(nDst*(isrc-1)+idst, :) = [srcCands{isrc}.point, dstCands{idst}.point, dstCands{idst}.type, labels(nDst*(isrc-1)+idst)];
        end
    end
    
    % save
    if (~isempty(cacheFile))
        save(cacheFile, 'features', 'distances', 'labels', 'jumps');
    end
end

% use only needed features (CVPR final)
if (isfield(options, 'featureIdx') && ~isempty(options.featureIdx))
    features = features(options.featureIdx, :);
end

% normalize features
if (isfield(options, 'featureNormMean'))
    nf = size(features, 2);
    %TODO strange: nf ~= nSrc * nDst when loaded from cache
    features = features - repmat(options.featureNormMean(:), [1, nf]);
    features = features ./ repmat(options.featureNormStd(:), [1, nf]);
%     features = features - repmat(options.featureNormMean(:), [1, nSrc * nDst]);
%     features = features ./ repmat(options.featureNormStd(:), [1, nSrc * nDst]);
end
