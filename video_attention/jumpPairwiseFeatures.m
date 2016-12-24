function [features, labels, jumps] = jumpPairwiseFeatures(srcFr, srcPts, dstFr, dstPts, dstTp, options)
% creates pairwise jump features
%
% [features, labels, jumps] = jumpPairwiseFeatures(srcFr, srcPts, dstFr, dstPts, dstTp, options)
%
% INPUT
%   srcFr, dstFr    source / destination frames, see preprocessFrames for
%                   details
%   srcPts, dstPts  source / destination points, [npt X 2]
%   dstTp           destination point type, 6 for gaze (i.e. ground truth)
%   options         options, used fields
%       .useLabel       if true labels will be creates (+1 if agree with
%                       ground truth, -1 if not
%       .dstGroundTruth ground truth map in destination frame. Used only if
%                       'useLabel' is true
%       .gazeThreshold  allowed deviation from true point
% OUTPUT
%   features        
%   labels      
%   jumps       array of jumps, [nsrc * ndst X 6], where each row is a
%               jump: [src_x, src_y, dst_x, dst_y, dst_type, label]

fvLen = 32; % 2*15+2
nSrc = size(srcPts, 1);
nDst = size(dstPts, 1);
features = zeros(fvLen, nSrc * nDst);
labels = zeros(1, nSrc * nDst);
jumps = zeros(nSrc * nDst, 6);

% find destination ground truth points
if (isfield(options, 'useLabel') && options.useLabel)
    rects = denseMap2Rects(options.dstGroundTruth, options.posPer, 0);
    gazeTh = options.gazeThreshold * srcFr.height;
    if (~isempty(rects))
        gtPts = rects(:, [1, 2]) + rects(:, [3, 4]) / 2;
    else
        gtPts = [];
    end
end

for isrc = 1:nSrc
    for idst = 1:nDst
        features(:, nDst*(isrc-1)+idst) = jumpFeaturePoint(srcFr, srcPts(isrc, :), dstFr, dstPts(idst, :), options);
        
        % set label
        if (isfield(options, 'useLabel') && options.useLabel)
            if (dstTp(idst) == 6) % destination is gaze
                labels(nDst*(isrc-1)+idst) = 1;
            else
                if (isempty(gtPts))
                    labels(nDst*(isrc-1)+idst) = -1;
                else
                    D = pdist2(gtPts, dstPts(idst, :), 'euclidean');
                    if (min(D) <= gazeTh)
                        labels(nDst*(isrc-1)+idst) = 1;
                    else
                        labels(nDst*(isrc-1)+idst) = -1;
                    end
                end
            end
        end
        
        % prepare jumps
        jumps(nDst*(isrc-1)+idst, :) = [srcPts(isrc, :), dstPts(idst, :), dstTp(idst), labels(nDst*(isrc-1)+idst)];
    end
end
