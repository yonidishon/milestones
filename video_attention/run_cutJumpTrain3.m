% run_cutJumpTrain3

%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
facesFile = fullfile(uncVideoRoot, '00_faces.mat');
humansFile = fullfile(uncVideoRoot, '00_humans.mat');
saliencyFile = fullfile(uncVideoRoot, '00_saliency.mat');
motionFile = fullfile(uncVideoRoot, '00_motion.mat');
gazeFile = fullfile(uncVideoRoot, '00_gaze.mat');
modelFile = fullfile(uncVideoRoot, '00_trained_model.mat');
saliencyRoot = fullfile(diemDataRoot, 'cuts_saliency');
motionRoot = fullfile(diemDataRoot, 'cuts_motion');
gazeRoot = fullfile(diemDataRoot, 'cuts_gaze');

cutTh = 10; % frames to consider cuts as the same
cutBefore = -1; % frames to sample before cut
cutAfter = 15; % frames to sample after cut

% options.nonMaxSuprRad = 2;
options.humanTh = 1;
options.humanMinSzRat = 0.3; % filter human smaller that this of maximum
options.humanMidSz = 0.3; % below this size (X height) create one candidate
options.humanTrackRat = 0.7; % tracks with ration of human bounding box
options.useLabel = true; % true if the label of feature should be calculated
options.candCovScale = 1;
options.motionScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.saliencyScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.sigmaScale = 0.5;
options.nSample = 10; % number of source samples
options.posPer = 0.1; % upper persentage for source sampling
options.negPer = 0.3; % lower threshold for source sampling
options.motionTh = 2; % optical flow below this not used
options.topCandsNum = 5;
options.topCandsUse = 5; % number of tracked candidates
options.minTrackSize = 20; % minimum size of the side of tracking rectangle
options.gazeThreshold = 10;

useCenter = true;

cache.root = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfile(cache.root, '00_features');
cache.renew = false; % use in case the preprocessing mechanism updated

% training set
% trainIdx = [28, 34, 41, 65];
trainIdx = 1:84;
calcIdx = 1:84;
% calcIdx = 34;
trainFeatureNum = 100000;
featureLen = 32; % TODO

preCalc = false;
trainModel = true;

s = load(facesFile);
videos = s.videos;
faces = s.faces;
s = load(humansFile);
humans = s.humans;
s = load(saliencyFile);
saliency = s.saliency;
s = load(motionFile);
motion = s.motion;
s = load(gazeFile);
gaze = s.gaze;
clear s;

%% gather features
if (preCalc)
    features = [];
    labels = [];
    % curFeat = 1;
    totalFeatures = zeros(length(calcIdx), 1);
    posFeatures = zeros(length(calcIdx), 1);
    negFeatures = zeros(length(calcIdx), 1);
    
    for i = 1:length(calcIdx)
        % reset features
        clear features labels;
        features = [];
        labels = [];
        
        iv = calcIdx(i);
        fprintf('Extracting features from %s... ', videos{iv}); tic;
        
        featureFile = fullfile(cache.featureRoot, sprintf('%s.mat', videos{iv}));
        if (~cache.renew && exist(featureFile, 'file')) % skip existing
            continue;
        end
        nc = length(faces{iv}.cuts);
        
        % load gaze, motion, saliency
        s = load(fullfile(gazeRoot, sprintf('%s.mat', videos{iv})));
        gaze = s.gaze;
        clear s;
        s = load(fullfile(saliencyRoot, sprintf('%s.mat', videos{iv})));
        saliency = s.saliency;
        clear s;
        s = load(fullfile(motionRoot, sprintf('%s.mat', videos{iv})));
        motion = s.motion;
        clear s;
        
        for ic = 1:nc
            if ((faces{iv}.cuts(ic) + cutBefore >= 3) && (faces{iv}.cuts(ic) + cutAfter <= faces{iv}.length) && ~isempty(gaze.before{ic}) && ~isempty(gaze.after{ic}) && ~isempty(saliency.before{ic}) && ~isempty(saliency.after{ic}))
                % source frame
                [srcPts, srcScore, ~] = sourceCands(gaze.before{ic}.gazeProb, options);
                
                % destination frame
                maps = cat(3, motion.after{ic}.magnitude, saliency.after{ic}.saliency, gaze.after{ic}.gazeProb);
                [cands, dstPts, dstScore, dstType] = jumpCandidates(faces{iv}.after{ic}, humans{iv}.after{ic}, maps, options);
                
                % prepare frames
                srcFr.width = faces{iv}.width;
                srcFr.height = faces{iv}.height;
                srcFr.faces = faces{iv}.before{ic};
                srcFr.poselet_hit = humans{iv}.before{ic};
                srcFr.ofx = motion.before{ic}.ofx;
                srcFr.ofy = motion.before{ic}.ofy;
                srcFr.saliency = saliency.before{ic}.saliency;
                
                dstFr.width = faces{iv}.width;
                dstFr.height = faces{iv}.height;
                dstFr.faces = faces{iv}.after{ic};
                dstFr.poselet_hit = humans{iv}.after{ic};
                dstFr.ofx = motion.after{ic}.ofx;
                dstFr.ofy = motion.after{ic}.ofy;
                dstFr.saliency = saliency.after{ic}.saliency;
                
                % features
                [f, l] = jumpPairwiseFeatures(srcFr, srcPts, dstFr, dstPts, dstType, options);
                clear srcFr dstFr;
                features = [features, f];
                labels = [labels, l];
            end
        end
        
        % save features
        totalFeatures(i) = length(labels);
        posFeatures(i) = sum(labels == 1);
        negFeatures(i) = sum(labels == -1);
        
        save(featureFile, 'features', 'labels', 'options');
        fprintf('%d features in %f sec\n', totalFeatures(i), toc);
    end
    
    % save results
    save(fullfile(cache.featureRoot, '00_total.mat'), 'options', 'trainIdx', 'totalFeatures', 'posFeatures', 'negFeatures');
end

%% train model
if (trainModel)
    % choose features
    total = load(fullfile(cache.featureRoot, '00_total.mat'));
    smpPos = zeros(size(total.totalFeatures));
    smpPos(trainIdx) = total.posFeatures(trainIdx);
    smpNeg = zeros(size(total.totalFeatures));
    smpNeg(trainIdx) = total.negFeatures(trainIdx);
    pf = sum(smpPos);
    nf = sum(smpNeg);
    nfeat = min([pf, nf, trainFeatureNum/2]); % number of +/- to sample

    smpPos = floor(smpPos .* (nfeat / pf) ./ 2) .* 2;
    smpNeg = floor(smpNeg .* (nfeat / nf) ./ 2) .* 2;
    
    posFeat = zeros(sum(smpPos), featureLen);
    negFeat = zeros(sum(smpNeg), featureLen);
    posL = ones(sum(smpPos), 1);
    negL = -1 .* ones(sum(smpNeg), 1);
    curPos = 1;
    curNeg = 1;

    fprintf('Training RF on %d positives and %d negatives...\n', sum(smpPos), sum(smpNeg)); tic;

    for i = 1:length(trainIdx)
        iv = trainIdx(i);
        fl = load(fullfile(cache.featureRoot, sprintf('%s.mat', videos{iv})));
        posIdx = find(fl.labels == 1);
        negIdx = find(fl.labels == -1);
        
        % sample positives
        if (~isempty(posIdx) && smpPos(iv) > 0)
            if (smpPos(iv) < length(posIdx))
                p = randperm(length(posIdx), smpPos(iv));
                posFeat(curPos:curPos+smpPos(iv)-1, :) = fl.features(:, posIdx(p))';
            else % take all
                posFeat(curPos:curPos+smpPos(iv)-1, :) = fl.features(:, posIdx)';
            end
            curPos = curPos + smpPos(iv);
        end
        
        % sample negatives
        if (~isempty(negIdx) && smpNeg(iv) > 0)
            if (smpNeg(iv) < length(negIdx))
                p = randperm(length(negIdx), smpNeg(iv));
                negFeat(curNeg:curNeg+smpNeg(iv)-1, :) = fl.features(:, negIdx(p))';
            else % take all
                negFeat(curNeg:curNeg+smpNeg(iv)-1, :) = fl.features(:, negIdx)';
            end
            curNeg = curNeg + smpNeg(iv);
        end
    end
    
    posFeat(isnan(posFeat)) = 0;
    negFeat(isnan(negFeat)) = 0;
    rf = regRF_train([posFeat; negFeat], [posL; negL]);
    save(modelFile, 'rf', 'options', 'trainIdx', '-v7.3');
    
    fprintf('\t...%f sec\n', toc);
end
