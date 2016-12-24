% run_jumpFrameCRCNS
% tests jump-track model without tracking. Jumps in every frame

%% settings
modelRoot = fullfile(diemDataRoot, 'video_unc');
uncVideoRoot = fullfile(crcnsOrigRoot, 'video_unc');
gazeDataRoot = fullfile(crcnsOrigRoot, 'gaze');
modelFile = fullfile(modelRoot, '00_trained_model_validation_v5_4.mat'); % model from train set (41 videos)
visRoot = fullfile(crcnsOrigRoot, 'vis_jump', 'v5_4');

evalOp.isDense = false;
evalOp.sigma = 10;
evalOp.pThresh = 0;

useCenter = true;
jumpType = 'cut';
measures = {'chisq', 'auc', 'cc', 'nss'};
methods = {'proposed', 'self', 'center', 'GBVS', 'PQFT', 'Hou'};

cutAfter = 1; % frames to update the map after cut

testIdx = 44; frames = {50:1400};

% caching
cache.frameRoot = fullfile(crcnsOrigRoot, 'cache');
cache.renew = false; % use in case the preprocessing mechanism updated

% visualization
cmap = jet(256);
colors = [1 0 0;
    0 1 0;
    0 0 1;
    1 1 0;
    1 0 1;
    0 1 1];

%% prepare
videos = videoListLoad(crcnsOrigRoot);
s = load(modelFile);
rf = s.rf;
options = s.options;
clear s;

param = loadDefaultParams;
param.inputDir = []; % to use video reader
param.debug = 0;
param.scoreTh = 6; % normal: 2, failure: 5...10
param.minSize = 1/10;

% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

nt = length(testIdx);
sim = cell(nt, 1);

%% run
for i = 1:nt
    iv = testIdx(i);
%     it = find(testSetIdx == iv);
    fprintf('Testing on %s...\n', videos{iv}); tic;

    if (isunix) % use matlab video reader on Unix
        param.videoReader = VideoReaderMatlab(fullfile(uncVideoRoot, sprintf('%s.mat', videos{iv})));
    else
        param.videoReader = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
    end
    m = param.videoReader.Height;
    n = param.videoReader.Width;
    videoLen = param.videoReader.numberOfFrames;
    
    % results
    nfr = length(frames{i});
    cands = cell(nfr, 1); % candidates per frame, rect and score inside
    predMaps = zeros(m, n, nfr); % dense maps
    
    frId = 1; % next processed frame index
    
    % jump to the first frame from center
    fprintf('Bootstrapping at frame %d\n', frames{i}(frId));
    srcCands = sourceCandidates([], [], options); % dummy source at center
    % v2
%     srcCands = {struct('point', [n/2, m/2], 'score', 1)}; % dummy source at center
    dstCands = jumpPerform(srcCands, frames{i}(frId), frames{i}(frId), param, options, gbvsParam, ofParam, poseletModel, rf, cache);
    cands{frId} = candidatesTopSelect(dstCands, options);
    frId = 2;
    
    while (frId <= nfr) % till we are inside frames range
        fprintf('Performing jump at frame %d\n', frames{i}(frId));
        
        % jump to the current frame
        dstCands = jumpPerform(cands{frId-1}, frames{i}(frId)-1, frames{i}(frId), param, options, gbvsParam, ofParam, poseletModel, rf, cache);
        fprintf('\tJump produced %d candidates\n', length(dstCands));
        
        % adjust jump
        if (frId + cutAfter > nfr)
            frh = frId:nfr;
        else
            frh = frId:frId + cutAfter - 1;
        end

        % select candidates
%         dstCands = candidatesTopSelect(dstCands, options);
        
        cands{frId} = dstCands;
        predMaps(:,:,frId) = candidate2map(dstCands, [n, m]);
        
        frId = frId+1;
    end
    
    % visualize
    indFr = 1:size(predMaps, 3);
    sim{i} = zeros(length(methods), length(measures), length(indFr));
    vw = VideoWriter(fullfile(visRoot, sprintf('%s.avi', videos{iv})), 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
    open(vw);

    % load gaze data
    s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videos{iv})));
    gazeData = s.gaze.data;
    clear s;

    try
        for ifr = 1:length(indFr)
            fr = preprocessFrames(param.videoReader, frames{i}(indFr(ifr)), gbvsParam, ofParam, poseletModel, cache);
            [sim{i}(:,:,ifr), outMaps] = similarityFrame2(predMaps(:,:,indFr(ifr)), gazeData{frames{i}(indFr(ifr))}, gazeData(frames{i}([1:indFr(ifr)-1, indFr(ifr)+1:end])), measures, ...
                'self', ...
                struct('method', 'center', 'cov', [(n/16)^2, 0; 0, (n/16)^2]), ...
                struct('method', 'saliency_GBVS', 'map', fr.saliency), ...
                struct('method', 'saliency_PQFT', 'map', fr.saliencyPqft), ...
                struct('method', 'saliency_Hou', 'map', fr.saliencyHou));
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
nmeas = length(measures);
for im = 1:length(measures)
    meanChiSq = nan(nt, length(methods));
    for i = 1:nt
        for j = 1:length(methods)
            chiSq = sim{i}(j,im,:);
            meanChiSq(i, j) = mean(chiSq(~isnan(chiSq)));
        end
    end
    
    ind = find(~isnan(meanChiSq(:,1)));
    meanChiSq = meanChiSq(ind, :);
    meanMeas = mean(meanChiSq, 1);
    lbl = videos(testIdx(testSubset(ind)));
    
    % add dummy if there is only one test
    if (size(meanChiSq, 1) == 1), meanChiSq = [meanChiSq; zeros(1, nmeas)]; end;
    
    figure, bar(meanChiSq);
    imLabel(lbl, 'bottom', -90, {'FontSize',8, 'Interpreter', 'None'});
    ylabel(measures{im});
    title(sprintf('Mean %s', mat2str(meanMeas, 2)));
    legend(methods);
    
    print('-dpng', fullfile(visRoot, sprintf('overall_%s.png', measures{im})));
end
