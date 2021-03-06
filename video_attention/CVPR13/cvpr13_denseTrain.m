% cvpr13_denseTrain
clear options;

%% settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% v7    dense training and testing for CVPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
% visRoot = fullfile(diemDataRoot, 'vis_jump', 'train_v5_3');
modelFile = fullfile(uncVideoRoot, '00_trained_model_cvpr13_v7_full.mat');

% options.nonMaxSuprRad = 2;
options.useCenter = 2; % 0 - none, 1 - always, 2 - only if no close
options.humanTh = 1;
options.humanMinSzRat = 0.3; % filter human smaller that this of maximum
options.humanMinSz = 0.15; % filter human smaller that this of frame height
options.humanMidSz = 0.4; % below this size (X height) create one candidate
options.humanTrackRat = 0.7; % tracks with ration of human bounding box
options.candCovScale = 1;
options.motionScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.saliencyScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.contrastScales = [2, 4, 8]; % 5x5, 9x9, 17x17 windows
options.sigmaScale = 0.5;
options.nSample = 10; % number of source samples
% options.posPer = 0.1; % upper persentage for source sampling, random
options.posPer = 0.03; % upper persentage for source sampling, rect
options.negPer = 0.3; % lower threshold for source sampling
options.motionTh = 2; % optical flow below this not used
options.topCandsNum = 5;
options.topCandsUse = 5; % number of tracked candidates
options.minTrackSize = 20; % minimum size of the side of tracking rectangle
% pairwize jump features
options.useLabel = true; % true if the label of feature should be calculated
options.distType = 'euc';
options.gazeThreshold = 0.2; % distance to gaze within it the candidate concidered as good (Euclidian)
% options.gazeThreshold = 5; % distance to gaze within it the candidate concidered as good (Mahalanobis)
options.rectSzTh = 0; % minimal size for gaze candidate
% options.featureIdx = 21:42; % v5 destination + distance
% options.featureIdx = 21:40; % v5_3 destination
options.featureIdx = 1:22; % v5_3 destination
% options.featureIdx = [1, 16, 20, 21:42]; % v5_4 destination, distance, source type
% training
options.rfType = 'class'; % 'class' for classification, 'reg' for pseudo-regression, 'reg-dist' for regression on distance
options.sourceCandScoreTh = 0.3;

sourceType = 'rect'; % 'random', 'rect', or 'cand'
jumpType = 'cut+neg'; % 'cut' or 'gaze_jump'
denseCandStep = 10;

visJumps = false; % the jumps will be visualized

% cache settings
cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfileCreate(cache.root, '00_features_v7');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = false; % use in case the feature extraction is updated

% gaze settings
gazeParam.pointSigma = 10;

% training and testing settings
% trainIdx = [5,8,9,10,13,16,18,19,24,30,32,34,35,36,37,40,42,43,44,47,48,49,53,56,57,58,60,62,64,65,69,70,71,72,74,76,78,79,80,82,83];
% testIdx = [1,3,6,7,11,12,14,15,17,20,21,22,23,25,26,27,28,29,31,33,38,39,41,45,46,50,51,52,54,55,59,61,63,66,67,68,73,75,77,81,84];
trainIdx = [1,3,5,7,9,13,17:33,35:41,43,45:47,49:52,56:58,60:69,71:73,75:82];
% trainIdx = [21, 39];

% precalcSubset = [12, 16]; trainSubset = [12, 16];
% precalcSubset = 25:length(trainIdx); trainSubset = [];
% precalcSubset = 1:10; trainSubset = 1:10;
precalcSubset = 1:length(trainIdx); trainSubset = 1:length(trainIdx);

trainFeatureNum = 100000;
negPosRatio = 5;
featureLen = length(options.featureIdx);

%% prepare
videos = videoListLoad(diemDataRoot, 'DIEM');
nv = length(videos);

% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

% output
if (visJumps && exist('visRoot', 'var'))
    cmap = jet(256);
    if (~exist(visRoot, 'dir'))
        mkdir(visRoot);
    end
end
    
%% gather features
if (~isempty(precalcSubset))
    calcIdx = trainIdx(precalcSubset);
    totalFeatures = zeros(nv, 1);
    posFeatures = zeros(nv, 1);
    negFeatures = zeros(nv, 1);
    
    for i = 1:length(calcIdx)
        % reset features
        clear features labels;
        features = [];
        labels = [];
        distances = [];
        
        iv = calcIdx(i);
        fprintf('Extracting features from %s... ', videos{iv}); tic;
        
        featureFile = fullfile(cache.featureRoot, sprintf('%s.mat', videos{iv}));
%         if (~cache.renewFeatures && exist(featureFile, 'file')) % skip existing
%             s = load(featureFile, 'labels');
%             totalFeatures(iv) = length(s.labels);
%             posFeatures(iv) = sum(s.labels == 1);
%             negFeatures(iv) = sum(s.labels == -1);
%             clear s;
%             fprintf('(in cache)\n');
%             continue;
%         end
        
        vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
        m = vr.Height;
        n = vr.Width;
        videoLen = vr.numberOfFrames;
        [jumpFrames, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType);
        nc = length(jumpFrames);
        
        % load gaze data
        s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videos{iv})));
        gazeParam.gazeData = s.gaze.data;
        clear s;

        % prepare visualisation
        if (visJumps && exist('visRoot', 'var'))
            vw = VideoWriter(fullfile(visRoot, sprintf('%s.avi', videos{iv})), 'Motion JPEG AVI');
            open(vw);
        end
        
        for ic = 1:nc
            if ((jumpFrames(ic) + before >= 3) && (jumpFrames(ic) + after < videoLen) && (jumpFrames(ic) + after < length(gazeParam.gazeData)))
                % preprocess frames
                srcFr = preprocessFrames(vr, jumpFrames(ic)+before, gbvsParam, ofParam, poseletModel, cache);
                dstFr = preprocessFrames(vr, jumpFrames(ic)+after, gbvsParam, ofParam, poseletModel, cache);

                % source candidates
                srcGazeMap = points2GaussMap(gazeParam.gazeData{jumpFrames(ic)+before}', ones(1, size(gazeParam.gazeData{jumpFrames(ic)+before}, 1)), 0, [n, m], gazeParam.pointSigma);
                srcCands = sourceCandidates(srcFr, srcGazeMap, options, sourceType);
%                 srcCands = denseCandidates(srcFr, denseCandStep);
                
                % destination candidates
                dstCands = denseCandidates(dstFr, denseCandStep);
                
                % features
                dstGazeMap = points2GaussMap(gazeParam.gazeData{jumpFrames(ic)+after}', ones(1, size(gazeParam.gazeData{jumpFrames(ic)+after}, 1)), 0, [n, m], gazeParam.pointSigma);
                options.dstGroundTruth = dstGazeMap;
                options.dstGroundTruthPts = gazeParam.gazeData{jumpFrames(ic)+after};
                [f, d, l, jumps] = jumpPairwiseFeatures6(srcFr, srcCands, dstFr, dstCands, options, cache);
%                 [f, d, l, jumps] = jumpPairwiseFeatures2(srcFr, srcCands, dstFr, dstCands, options);
%                 [f, l, jumps] = jumpPairwiseFeatures(srcFr, srcPts, dstFr, dstPts, dstType, options);
                clear srcFr dstFr;
                
                % visualize
                if (visJumps)
                    try
                        srcGazeMap = points2GaussMap(gazeParam.gazeData{jumpFrames(ic)+before}', ones(1, size(gazeParam.gazeData{jumpFrames(ic)+before}, 1)), 0, [n, m], gazeParam.pointSigma);
                        outfr = visJumpsSbs(vr, jumpFrames(ic)+before, srcGazeMap, jumpFrames(ic)+after, dstGazeMap, jumps, cmap);
                        writeVideo(vw, outfr);
                    catch me
                        fprintf('ERROR: in writing video %s: %s\n', videos{iv}, me.message);
                    end
                end
                
                % store
                features = [features, f];
                labels = [labels, l];
                distances = [distances, d];
            end
        end
        
        % save features
        totalFeatures(iv) = length(labels);
        posFeatures(iv) = sum(labels == 1);
        negFeatures(iv) = sum(labels == -1);
        
        save(featureFile, 'features', 'labels', 'distances', 'options');
        
        if (visJumps)
            close(vw);
        end
        
        fprintf('%d features in %f sec\n', totalFeatures(iv), toc);
    end
    
    % save results
    save(fullfile(cache.featureRoot, '00_total.mat'), 'options', 'trainIdx', 'precalcSubset', 'trainSubset', 'totalFeatures', 'posFeatures', 'negFeatures');
end

%% train model
if (~isempty(trainSubset))
    trIdx = trainIdx(trainSubset);
    % choose features
    total = load(fullfile(cache.featureRoot, '00_total.mat'));
    smpPos = zeros(size(total.totalFeatures));
    smpPos(trIdx) = total.posFeatures(trIdx);
    smpNeg = zeros(size(total.totalFeatures));
    smpNeg(trIdx) = total.negFeatures(trIdx);
    pf = sum(smpPos);
    nf = sum(smpNeg);
    nfeatPos = min(pf, trainFeatureNum/(1+negPosRatio)); % number of +
    nfeatNeg = min([nf, negPosRatio*trainFeatureNum/(1+negPosRatio), negPosRatio*nfeatPos]); % number of -
%     nfeat = min([pf, nf, trainFeatureNum/2]); % number of +/- to sample

    smpPos = floor(smpPos .* (nfeatPos / pf) ./ 2) .* 2;
    smpNeg = floor(smpNeg .* (nfeatNeg / nf) ./ 2) .* 2;
    
    posFeat = zeros(sum(smpPos), featureLen);
    negFeat = zeros(sum(smpNeg), featureLen);
    posDist = zeros(sum(smpPos), 1);
    negDist = zeros(sum(smpNeg), 1);
    posL = ones(sum(smpPos), 1);
    negL = -1 .* ones(sum(smpNeg), 1);
    curPos = 1;
    curNeg = 1;

    fprintf('Training RF on %d positives and %d negatives...\n', sum(smpPos), sum(smpNeg)); tic;

    for i = 1:length(trIdx)
        iv = trIdx(i);
        fl = load(fullfile(cache.featureRoot, sprintf('%s.mat', videos{iv})));
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
    
    % normalize features
    feat = [posFeat; negFeat];
    nfeat = size(feat, 1);
    options.featureNormMean = mean(feat, 1);
    feat = feat - repmat(options.featureNormMean, [nfeat, 1]);
    options.featureNormStd = std(feat, 0, 1);
    feat = feat ./ repmat(options.featureNormStd, [nfeat, 1]);
    feat(isnan(feat)) = 0;
    
    % train random forest
    if (strcmp(options.rfType, 'reg'))
        rf = regRF_train(feat, [posL; negL]);
    elseif (strcmp(options.rfType, 'class'))
        negL(:) = 0;
        rf = classRF_train(feat, [posL; negL]);
    elseif (strcmp(options.rfType, 'reg-dist'))
        dist = [posDist; negDist];
        md = max(dist(~isinf(dist)));
        dist(isinf(dist)) = md;
        rf = regRF_train(feat, dist);
    else
        error('Unsupported RF type: %s', options.rfType);
    end
    
    save(modelFile, 'rf', 'options', 'trainIdx', '-v7.3');
    
    fprintf('\t...%f sec\n', toc);
end
