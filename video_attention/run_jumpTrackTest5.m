% run_jumpTrackTest5
% tests jump-track version 5 on frame range
% tracks candidate points (with rectangles around them) till failure
% can revive dead tracks

%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
modelFile = fullfile(uncVideoRoot, '00_trained_model_validation_v5.mat'); % model from train set (41 videos)
visRoot = fullfile(diemDataRoot, 'vis_jumptrack', 'track_v5_1');

evalOp.isDense = false;
evalOp.sigma = 10;
evalOp.pThresh = 0;

jumpType = 'gaze_jump'; % 'cut' or 'gaze_jump'
measures = {'chisq', 'auc'};

cutBefore = -1; % frames to use before cut
cutAfter = 6; % frames to update the map after cut
predRectTh = 0.03; % percentage for rectangle tracking
rect2DenseScale = 0.5; % in conversion of rectangles to map
rectSz = [20 20]; % rectangle around each candidate to track
nTrackRect = 5; % number of rectangles to track
nTrackDeadTh = 3; % if this number of tracks are dead - jump

testIdx = 28; frames = {50:3550}; % bullet witch, entire
% testIdx = 39; frames = {10:2500}; % alice, entire
% testIdx = 40; frames = {2345:2638}; % ice age
% testIdx = 40; frames = {2395:2638}; % ice age
% testIdx = 40; frames = {2455:2638}; % ice age
% testIdx = 40; frames = {2180:2750}; % ice age
% testIdx = 34; frames = {1772:2331}; % harry potter
% testIdx = 34; frames = {2041:2331}; % harry potter
% testIdx = 13; frames = {36:519};
% testIdx = 21; frames = {802:1139};
% testIdx = 21; frames = {619:1030}; % chilli plasters, rice + dialog
% testIdx = 21; frames = {619:630};
% testIdx = 65; frames = {915:1680}; % barcelona sport
% testIdx = 41; frames = {1600:2100}; % quantum of solace
% testIdx = 23; frames = {300:1300}; % coral reef

% caching
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.renew = false; % use in case the preprocessing mechanism updated

cmap = jet(256);
colors = [1 0 0;
    0 1 0;
    0 0 1;
    1 1 0;
    1 0 1;
    0 1 1];

%% prepare
videos = videoListLoad(diemDataRoot, 'DIEM');
s = load(modelFile);
rf = s.rf;
options = s.options;
options.useLabel = false;
options.featureIdx = 21:42; % v5 destination + distance, HACK
clear s;

param = loadDefaultParams;
param.inputDir = []; % to use video reader
param.debug = 0;
param.scoreTh = 6; % normal: 2, failure: 5...10
param.minSize = 1/10;

% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

%% run
nt = length(testIdx);
sim = cell(nt, 1);

for i = 1:nt
    iv = testIdx(i);
%     it = find(testSetIdx == iv);
    fprintf('Testing on %s...\n', videos{iv}); tic;

    param.videoReader = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
    m = param.videoReader.Height;
    n = param.videoReader.Width;
    videoLen = param.videoReader.numberOfFrames;
    
    % load jump frames
    [cuts, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType);
    
    % find the cuts
    nfr = length(frames{i});
    cutIdx = find(cuts >= frames{i}(1) & cuts <= frames{i}(end));
    nc = length(cutIdx);
    cutFrameIdx = zeros(nc, 1);
    for j = 1:length(cutIdx)
        cutFrameIdx(j) = find(frames{i} == cuts(cutIdx(j)));
    end
    
    % results
    cands = cell(nfr, 1); % candidates per frame, rect and score inside
    predMaps = zeros(m, n, nfr); % dense maps
    
    frId = 1; % next processed frame index
    if (cutFrameIdx(1) == 1) % start at cut
        doTrack = false;
        doJump = true;
    else
        doTrack = true;
        doJump = false;
    end
    doRevive = false;
    
    % jump to the first frame from center
    fprintf('Bootstrapping at frame %d\n', frames{i}(frId));
    srcCands = sourceCandidates([], [], options); % dummy source at center
    % v2
%     srcCands = {struct('point', [n/2, m/2], 'score', 1)}; % dummy source at center
    dstCands = jumpPerform(srcCands, frames{i}(frId), frames{i}(frId), param, options, gbvsParam, ofParam, poseletModel, rf, cache);
    cands{frId} = candidatesTopSelect(dstCands, options);
    frId = 2;
    
    while (frId <= nfr) % till we are inside frames range
        if (doJump) % jump
            fprintf('Performing jump at frame %d\n', frames{i}(frId));
            
            % v.2: allows jumps from every frame to every frame.
            %   jumps only cutAfter distance, while the map is held
            if ((frames{i}(frId) - 3 > 1) && (frames{i}(frId) + cutAfter <= videoLen)) % can jump
                % jump
                dstCands = jumpPerform(cands{frId-1}, frames{i}(frId)-1, frames{i}(frId)+cutAfter, param, options, gbvsParam, ofParam, poseletModel, rf, cache);
                fprintf('\tJump produced %d candidates\n', length(dstCands));

                % adjust jump
                if (frId + cutAfter > nfr)
                    frh = frId:nfr;
                else
                    frh = frId:frId + cutAfter - 1;
                end

                % select candidates
                dstCands = candidatesTopSelect(dstCands, options);
                
                % hold till last frame
                mp = candidate2map(cands{frId-1}, [n, m]);
                for ifrh = 1:length(frh)-1
                    cands{frh(ifrh)} = cands{frId-1};
                    predMaps(:,:,frh(ifrh)) = mp;
                end
                fprintf('\tHeld map at %d->%d\n', frames{i}(frh(1)), frames{i}(frh(end-1)));

                cands{frh(end)} = dstCands;
                predMaps(:,:,frh(end)) = candidate2map(dstCands, [n, m]);
                fprintf('\tJump map used at frame %d\n', frames{i}(frh(end)));

                % follow with track
                doJump = false;
                doTrack = true;
                frId = frh(end)+1;
            else
                error('Cannot jump at frame %d', frames{i}(frId));
            end
        end
        
        % early stop
        if (frId >= nfr)
            break;
        end
        
        if (doTrack) % track
            doRevive = true;
            
            % tracking boundaries
            param.initFrame = frames{i}(frId-1); % start tracking from previous frame
            nextCutId = find(cuts > param.initFrame, 1, 'first');
            if (isempty(nextCutId) || frames{i}(end) < cuts(nextCutId)) % last shot - track to end
                param.finFrame = frames{i}(end);
            else
                param.finFrame = cuts(nextCutId);
            end
            
            ntfr = param.finFrame - param.initFrame + 1;
            nr = length(cands{frId-1}); % number of tracked candidates
            fprintf('Tracking %d candidates in frames %d->%d (%d frames)\n', nr, param.initFrame, param.finFrame, ntfr);
            
            tracking = zeros(ntfr, 4, nr);
            tracksValid = false(ntfr, nr);
            trackLen = zeros(nr, 1);
            trackIdx = 1:nr; % initially track all
            
            while (doRevive)
                for idxr = 1:length(trackIdx) % for every rectangle
                    ir = trackIdx(idxr);
                    fprintf('\tTracking rectangle #%d at %s (%d):', ir, mat2str(cands{frId-1}{ir}.trackRect), cands{frId-1}{ir}.type);
                    param.target0 = cands{frId-1}{ir}.trackRect;
                    [target, score, validIdx] = LOT_trackTillFailure(param);
                    tracking(validIdx,:,ir) = target;
                    tracksValid(validIdx,ir) = true;
                    trackLen(ir) = length(validIdx);
                    fprintf('%d frames\n', trackLen(ir));
                end
            
                % choose track with minimum length
                tfr = min(trackLen);
                trackingValid = tracking(1:tfr, :, :);
                
                % update candidates
                for jfr = 1:tfr
                    for ir = 1:nr % for every rectangle
                        cands{frId+jfr-1}{ir} = cands{frId+jfr-2}{ir}; % basis
                        oldCen = cands{frId+jfr-1}{ir}.trackRect([1,2]) + 0.5*cands{frId+jfr-1}{ir}.trackRect([3,4]);
                        newRect = trackingValid(jfr, :, ir);
                        newCen = newRect([1,2]) + 0.5*newRect([3,4]);
                        ptTr = newCen - oldCen;
                        cands{frId+jfr-1}{ir}.trackRect = newRect; % update rectangle
                        cands{frId+jfr-1}{ir}.point = cands{frId+jfr-1}{ir}.point + ptTr; % update point
                    end
                    
                    % dense map
                    predMaps(:,:,frId+jfr-1) = candidate2map(cands{frId+jfr-1}, [n, m]);
                end
                
                % update frame progress
                frId = frId + tfr;
                ntfr = ntfr - tfr;
                trackLen = trackLen - tfr;
                if (frId >= nfr)
                    break;
                end
                fprintf('Tracking stopped at %d\n', frames{i}(frId));
            
                % find dead tracks
                if (tfr < ntfr && frId > 2)
                    tracking = tracking(tfr+1:end, :, :);
                    tracksValid = tracksValid(tfr+1:end,:);
                    tracksDeadIdx = find(~tracksValid(1,:));
                    nTracksDead = length(tracksDeadIdx);
                    
                    % jump on catastrophy
                    if (nTracksDead >= nTrackDeadTh)
                        doRevive = false;
                        break;
                    end
                    
                    % revive
                    fprintf('Reviving %d tracks at frame %d\n', nTracksDead, frames{i}(frId));
                    
                    % new candidates
                    dstCands = jumpPerform(cands{frId-2}, frames{i}(frId)-2, frames{i}(frId)-1, param, options, gbvsParam, ofParam, poseletModel, rf, cache);
                    dstCands = candidatesTopSelect(dstCands, struct('topCandsUse', nTracksDead));
                    for idc = 1:nTracksDead
                        it = tracksDeadIdx(idc);
                        cands{frId-1}{it} = dstCands{idc};
                    end
                    
                    % prepare to track revived tracks
                    trackIdx = tracksDeadIdx;
                    param.initFrame = frames{i}(frId-1);
                else % jump if cannot revive
                    doRevive = false;
                end
            end
            
            % follow with jump
            doJump = true;
            doTrack = false;
        end
    end
    
    % visualize
%     simAUC{i} = visSideBySideCompareVideo(diemDataRoot, visVideoRoot, iv, frames{i}, predMaps(:,:,1:length(frames{i})), cands(1:length(frames{i})));
    % visualize
    indFr = 1:size(predMaps, 3);
    sim{i} = zeros(4, length(measures), length(indFr));
    vw = VideoWriter(fullfile(visRoot, sprintf('%s.avi', videos{iv})), 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
    open(vw);

    % load gaze data
    s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videos{iv})));
    gazeData = s.gaze.data;
    clear s;

    try
        for ifr = 1:length(indFr)
            fr = preprocessFrames(param.videoReader, frames{i}(indFr(ifr)), gbvsParam, ofParam, poseletModel, cache);
            [sim{i}(:,:,ifr), outMaps] = similarityFrame(predMaps(:,:,indFr(ifr)), gazeData{frames{i}(indFr(ifr))}, measures, ...
                'self', ...
                struct('method', 'center', 'cov', [(n/16)^2, 0; 0, (n/16)^2]), ...
                struct('method', 'saliency_GBVS', 'map', fr.saliency));
            outfr = renderSideBySide(fr.image, outMaps, colors, cmap, sim{i}(:,:,ifr));
            writeVideo(vw, outfr);
        end
    catch me
        close(vw);
        rethrow(me);
    end
    
    close(vw);

    fprintf('%f sec\n', toc);
end

%% visualize
nmeas = 4;
leg = {'proposed', 'self', 'center', 'GBVS'};
for im = 1:length(measures)
    meanChiSq = nan(nt, nmeas);
    for i = 1:nt
        for j = 1:nmeas
            chiSq = sim{i}(j,im,:);
            meanChiSq(i, j) = mean(chiSq(~isnan(chiSq)));
        end
    end
    
    ind = find(~isnan(meanChiSq(:,1)));
    meanChiSq = meanChiSq(ind, :);
    meanMeas = mean(meanChiSq, 1);
    lbl = videos(testIdx(ind));
    
    if (size(meanChiSq, 1) == 1), meanChiSq = [meanChiSq; zeros(1, nmeas)]; end;
    
    figure, bar(meanChiSq);
    imLabel(lbl, 'bottom', -90, {'FontSize',8, 'Interpreter', 'None'});
    ylabel(measures{im});
    title(sprintf('Mean %s', mat2str(meanMeas, 2)));
    legend(leg);
    
    print('-dpng', fullfile(visRoot, sprintf('overall_%s.png', measures{im})));
end
