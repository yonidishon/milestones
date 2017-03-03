function [ Xtrain, Ytrain ] = prepdata( total,trIdx,featureRoot,options)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    % choose features
    smpPos = zeros(size(total.totalFeatures));
    smpPos(trIdx) = total.posFeatures(trIdx);
    smpNeg = zeros(size(total.totalFeatures));
    smpNeg(trIdx) = total.negFeatures(trIdx);
    pf = sum(smpPos);
    nf = sum(smpNeg);
    nfeatPos = min(pf, options.trainFeatureNum/(1+options.negPosRatio)); % number of +
    nfeatNeg = min([nf, options.negPosRatio*options.trainFeatureNum/(1+options.negPosRatio),...
        options.negPosRatio*nfeatPos]); % number of -
%     nfeat = min([pf, nf, trainFeatureNum/2]); % number of +/- to sample

    smpPos = floor(smpPos .* (nfeatPos / pf) ./ 2) .* 2;
    smpNeg = floor(smpNeg .* (nfeatNeg / nf) ./ 2) .* 2;
    
    posFeat = zeros(sum(smpPos), options.featureLen);
    negFeat = zeros(sum(smpNeg), options.featureLen);
    posDist = zeros(sum(smpPos), 1);
    negDist = zeros(sum(smpNeg), 1);
    posL = ones(sum(smpPos), 1);
    negL = -1 .* ones(sum(smpNeg), 1);
    curPos = 1;
    curNeg = 1;

    fprintf('Training RF on %d positives and %d negatives...\n', sum(smpPos), sum(smpNeg)); tic;

    for i = 1:length(trIdx)
        iv = trIdx(i);
        fl = load(fullfile(featureRoot, sprintf('%s.mat', options.videos{iv})));
        posIdx = find(fl.labels == 1);
        negIdx = find(fl.labels == -1);
        
        % sample positives
        if (~isempty(posIdx) && smpPos(iv) > 0)
            if (smpPos(iv) < length(posIdx))
                p = randperm(length(posIdx));
                p = p(1:smpPos(iv));
                posFeat(curPos:curPos+smpPos(iv)-1, :) = fl.features(:, posIdx(p))';
                posDist(curPos:curPos+smpPos(iv)-1) = fl.distances(posIdx(p))';
            else % take all
                posFeat(curPos:curPos+smpPos(iv)-1, :) = fl.features(:, posIdx)';
                posDist(curPos:curPos+smpPos(iv)-1) = fl.distances(posIdx)';
            end
            curPos = curPos + smpPos(iv);
        end
        
        % sample negatives
        if (~isempty(negIdx) && smpNeg(iv) > 0)
            if (smpNeg(iv) < length(negIdx))
                p = randperm(length(negIdx));
                p = p(1:smpNeg(iv));
                negFeat(curNeg:curNeg+smpNeg(iv)-1, :) = fl.features(:, negIdx(p))';
                negDist(curNeg:curNeg+smpNeg(iv)-1) = fl.distances(negIdx(p))';
            else % take all
                negFeat(curNeg:curNeg+smpNeg(iv)-1, :) = fl.features(:, negIdx)';
                negDist(curNeg:curNeg+smpNeg(iv)-1) = fl.distances(negIdx)';
            end
            curNeg = curNeg + smpNeg(iv);
        end
    end
    
    posFeat(isnan(posFeat)) = 0;
    negFeat(isnan(negFeat)) = 0;
    
    feat = [posFeat; negFeat];
    
    if(options.norm)
        % normalize features    
        nfeat = size(feat, 1);
        tmpfeat = feat;
        tmpfeat(~isfinite(feat))=nan;
        tmpfeat(tmpfeat==0)=nan;
        options.featureNormMean = nanmean(tmpfeat, 1);
        feat = tmpfeat - repmat(options.featureNormMean, [nfeat, 1]);
        options.featureNormStd = nanstd(tmpfeat, 0, 1);
        feat = feat ./ (repmat(options.featureNormStd, [nfeat, 1])+eps());
        %feat=tmpfeat;
    end
    
    Ytrain = [posL; negL];
    Xtrain = feat;
end

