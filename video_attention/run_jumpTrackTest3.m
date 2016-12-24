% run_jumpTrackTest3
% tests jump-track simple version on frame range
% tracking in this version stops at some frame (when tracking fails)

%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
modelFile = fullfile(uncVideoRoot, '00_trained_model.mat');
gazeRoot = fullfile(diemDataRoot, 'cuts_gaze');
saliencyRoot = fullfile(diemDataRoot, 'cuts_saliency');
motionRoot = fullfile(diemDataRoot, 'cuts_motion');
facesFile = fullfile(uncVideoRoot, '00_faces.mat');
humansFile = fullfile(uncVideoRoot, '00_humans.mat');
visVideoRoot = fullfile(diemDataRoot, 'vis_jumptrack', 'RF_sbs_v2');

% for candidates
options.nonMaxSuprRad = 2;
options.humanTh = 1;
options.useLabel = false;
% for features
options.motionScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.saliencyScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.sigmaScale = 0.5;
options.nSample = 10;
options.posPer = 0.1;
options.negPer = 0.3;

evalOp.isDense = false;
evalOp.denseStep = 10;
evalOp.sigma = 10;
evalOp.pThresh = 0;

useCenter = true;

cutBefore = -1; % frames to use before cut
cutAfter = 6; % frames to update the map after cut
predRectTh = 0.03; % percentage for rectangle tracking
rect2DenseScale = 0.3; % in conversion of rectangles to map

% testIdx = 34; % 34 - Harry Potter, 39 - Alice, 1:84 - all, 40 - Ice Age
% frames = {942:1029};
% frames = {920:1029};
testIdx = 40;
% frames = {2132:2638};
frames = {2345:2638};
% testIdx = 13;
% frames = {36:519};
% testIdx = 21;
% frames = {802:1139};

% caching
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.renew = false; % use in case the preprocessing mechanism updated

%% prepare
s = load(facesFile);
videos = s.videos;
faces = s.faces;
s = load(humansFile);
humans = s.humans;
% s = load(fullfile(uncVideoRoot, '00_test_all.mat'));
s = load(fullfile(uncVideoRoot, '00_test_set1.mat'));
cutPredMaps = s.denseJumpsMap;
testSetIdx = s.testIdx;
s = load(modelFile);
rf = s.rf;

clear s;

param = loadDefaultParams;
param.inputDir = []; % to use video reader
param.debug = 0;
param.scoreTh = 6; % normal: 2, failure: 5...10

% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

%% run
simAUC = cell(length(testIdx), 1);

for i = 1:length(testIdx)
    iv = testIdx(i);
    it = find(testSetIdx == iv);
    fprintf('Testing on %s...\n', videos{iv}); tic;

    param.videoReader = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
    m = faces{iv}.height;
    n = faces{iv}.width;
    
    % find the cuts
    nfr = length(frames{i});
    cuts = faces{iv}.cuts;
    cutIdx = find(cuts >= frames{i}(1) & cuts <= frames{i}(end));
    nc = length(cutIdx);
    cutFrameIdx = zeros(nc, 1);
    for j = 1:length(cutIdx)
        cutFrameIdx(j) = find(frames{i} == cuts(cutIdx(j)));
    end
    
    predMaps = zeros(faces{iv}.height, faces{iv}.width, nfr);
    
    frId = 1;
    cutId = 0;
    useGaze = true; % for the first frame use gaze data
    if (cutFrameIdx(1) == 1) % start at cut
        cutId = 1;
        doTrack = false;
        doJump = true;
    else % track for gaze
        doTrack = true;
        doJump = false;
    end        
    
    if (useGaze)
        s = load(fullfile(gazeRoot, sprintf('%s.mat', videos{iv})));
        gaze = s.gaze;
        clear s;
    end
    % load motion, saliency
    s = load(fullfile(saliencyRoot, sprintf('%s.mat', videos{iv})));
    saliency = s.saliency;
    clear s;
    s = load(fullfile(motionRoot, sprintf('%s.mat', videos{iv})));
    motion = s.motion;
    clear s;

    
    while (frId <= nfr) % till we are inside frames range
        if (doJump) % jump
            fprintf('Performing jump at frame %d\n', frames{i}(frId));
            ic = cutIdx(cutId);
            
            % jump source frame
            if (useGaze)
                if (~isempty(gaze.before{ic}))
                    prevMap = gaze.before{ic}.gazeProb;
                else
                    warning('Gaze at frame %d is empty', frames{i}(frId));
                    prevMap = zeros(m, n);
                end
                useGaze = false;
            else
                prevMap = predMaps(:,:,frId - 1);
            end
            
            % v.2: allows jumps from every frame to every frame.
            %   jumps only cutAfter distance, while the map is held
            if ((frames{i}(frId) - 3 > 1) && (frames{i}(frId) + cutAfter <= faces{iv}.length)) % can jump
                [srcPts, srcScore, ~] = sourceCands(prevMap, options);
                nSrc = size(srcPts, 1);
                
                srcFr = preprocessFrames(param.videoReader, frames{i}(frId)-1, gbvsParam, ofParam, poseletModel, cache);
                dstFr = preprocessFrames(param.videoReader, frames{i}(frId)+cutAfter, gbvsParam, ofParam, poseletModel, cache);
                
                maps = cat(3, (dstFr.ofx.^2 + dstFr.ofy.^2), dstFr.saliency);
                [dstPts, dstScore, dstType] = jumpCands(dstFr.faces, dstFr.poselet_hit, useCenter, maps, options);
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

                % convert to dense map
                jumpMap = evalCutJump([n, m], {jumps}, evalOp);
                frh = frId:frId + cutAfter - 1;
                predMaps(:,:,frh) = repmat(prevMap, [1,1,length(frh)]);
                predMaps(:,:,frh(end)+1) = jumpMap;
                fprintf('Held previous map at %d->%d\n', frames{i}(frh(1)), frames{i}(frh(end)));
                fprintf('Jump map used at frame %d\n', frames{i}(frh(end))+1);
            else
                warning('Cannot jump at frame %d', frames{i}(frId));
            end
            
            % follow with track
            doJump = false;
            doTrack = true;
            frId = frh(end)+1;
        end
        
        if (doTrack) % track
            param.initFrame = frames{i}(frId);
            if (cutId == nc) % last cut - track to end
                param.finFrame = frames{i}(end);
            else
                param.finFrame = cuts(cutIdx(cutId + 1));
            end
            fprintf('Trying to track: %d->%d\n', param.initFrame, param.finFrame);
            
            ntfr = param.finFrame - param.initFrame + 1;
            
            rects = denseMap2Rects(predMaps(:,:,frId), predRectTh);
            nr = size(rects, 1);
            fprintf('\tIn jump at frame %d found %d rectangles\n', param.initFrame, nr);
            
            tracking = zeros(ntfr, 4, nr);
            trackLen = zeros(nr, 1);
            
            for ir = 1:nr % for every rectangle
                param.target0 = rects(ir, :);
                [target, score, validIdx] = LOT_trackTillFailure(param);
                tracking(validIdx,:,ir) = target;
                trackLen(ir) = length(validIdx);
            end
            
            % choose track with minimum length
            tfr = min(trackLen);
            tracking = tracking(1:tfr, :, :);
            
            % v2: warp the initial map
            predMapTrack = mapTransformTrack(predMaps(:,:,frId), tracking);
            
            % store the maps
            predMaps(:,:,frId:frId+tfr-1) = predMapTrack;
            
            % jump at following
            doJump = true;
            doTrack = false;
            frId = frId+tfr;
            
            if (tfr < ntfr) % jump because of failure
                fprintf('Tracking failed at frame %d\n', frames{i}(frId)-1);
            else % jump at cut
                cutId = cutId + 1;
            end
        end
    end
    
    % visualize
    simAUC{i} = visSideBySideCompareVideo(diemDataRoot, visVideoRoot, iv, frames{i}, predMaps);

    fprintf('%f sec\n', toc);
end
