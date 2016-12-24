% run_jumpValidationTest
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VERSIONS
% v4_43 - hard classification, Euclidian distance, normalization
% v5_1  - features with only destination and distance, hard classification, Euclidian distance, normalization
% v5_12 - jump from center, features with only destination and distance, hard classification, Euclidian distance, normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
visRoot = fullfile(diemDataRoot, 'vis_jump', 'test_v5_41');
modelFile = fullfile(uncVideoRoot, '00_trained_model_validation_v5_4.mat');

useCenter = true;
useDestGaze = false;
jumpType = 'gaze_jump'; % 'cut' or 'gaze_jump'
sourceType = 'rect';
measures = {'chisq', 'auc', 'cc', 'nss'};
methods = {'proposed', 'self', 'center', 'GBVS', 'PQFT', 'Hou'};

% cache settings
cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfile(cache.root, '00_features_v5');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = false; % use in case the feature extraction is updated

% gaze settings
gazeParam.pointSigma = 10;

% training and testing settings
trainIdx = [5,8,9,10,13,16,18,19,24,30,32,34,35,36,37,40,42,43,44,47,48,49,53,56,57,58,60,62,64,65,69,70,71,72,74,76,78,79,80,82,83];
testIdx = [1,3,6,7,11,12,14,15,17,20,21,22,23,25,26,27,28,29,31,33,38,39,41,45,46,50,51,52,54,55,59,61,63,66,67,68,73,75,77,81,84];

% testSubset = 1:length(trainIdx);
testSubset = 11;
jumpFromCenter = false; % use center as source
visVideo = true;

% visualization
cmap = jet(256);
colors = [1 0 0;
    0 1 0;
    0 0 1;
    1 1 0;
    1 0 1;
    0 1 1];

%% prepare
videos = videoListLoad(diemDataRoot, 'DIEM');
nv = length(videos);

% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

% load model
s = load(modelFile);
rf = s.rf;
options = s.options;
options.useLabel = false; % no need in label while testing
clear s;

nt = length(testSubset);
sim = cell(nt, 1);

vers = version('-release');
verNum = str2double(vers(1:4));

if (~exist(visRoot, 'dir'))
    mkdir(visRoot);
end

%% test
for i = 1:nt
    iv = testIdx(testSubset(i));
    fprintf('Processing %s... ', videos{iv}); tic;
    
    % prepare video
    if (isunix) % use matlab video reader on Unix
        vr = VideoReaderMatlab(fullfile(uncVideoRoot, sprintf('%s.mat', videos{iv})));
    else
        if (verNum < 2011)
            vr = mmreader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
        else
            vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
        end
    end
    m = vr.Height;
    n = vr.Width;
    videoLen = vr.numberOfFrames;
    param = struct('videoReader', vr);
    
    % load jump frames
    [jumpFrames, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType);
    nc = length(jumpFrames);
    
    % load gaze data
    s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videos{iv})));
    gazeParam.gazeData = s.gaze.data;
    clear s;

    videoLen = min(videoLen, length(gazeParam.gazeData));
    
    cands = cell(nc, 1); % candidates per frame
    predMaps = zeros(m, n, nc); % dense maps
    
    for ic = 1:nc
        if ((jumpFrames(ic) + before >= 3) && (jumpFrames(ic) + after <= videoLen))
            % source frame
            if (jumpFromCenter)
                srcCands = {struct('point', [n/2, m/2], 'score', 1)}; % dummy source at center
            else % jump from gaze map
                srcGazeMap = points2GaussMap(gazeParam.gazeData{jumpFrames(ic)+before}', ones(1, size(gazeParam.gazeData{jumpFrames(ic)+before}, 1)), 0, [n, m], gazeParam.pointSigma);
                [srcCands, ~, ~, ~] = sourceCandidates([], srcGazeMap, options, sourceType);
                % v2
%                 [srcPts, srcScore, ~] = sourceCands(srcGazeMap, options, sourceType);
%                 srcCands = cell(length(srcScore), 1);
%                 for isrc = 1:length(srcScore)
%                     srcCands{isrc} = struct('point', srcPts(isrc, :), 'score', srcScore(isrc));
%                 end
            end
            
            dstCands = jumpPerform(srcCands, jumpFrames(ic)+before, jumpFrames(ic)+after, param, options, gbvsParam, ofParam, poseletModel, rf, cache);
            predMaps(:,:,ic) = candidate2map(dstCands, [n, m]);
            cands{ic} = dstCands;
        end
    end
    
    % compare
    frames = jumpFrames + after;
    indFr = find(frames <= videoLen);
    
    % save
    save(fullfile(visRoot, sprintf('%s.mat', videos{iv})), 'frames', 'indFr', 'cands', 'predMaps');
    
    % visualize
    sim{i} = zeros(length(methods), length(measures), length(indFr));
    if (visVideo && verNum >= 2012)
        vw = VideoWriter(fullfile(visRoot, sprintf('%s.avi', videos{iv})), 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
        open(vw);
    end
    
    try
        for ifr = 1:length(indFr)
            fr = preprocessFrames(param.videoReader, frames(indFr(ifr)), gbvsParam, ofParam, poseletModel, cache);
            [sim{i}(:,:,ifr), outMaps] = similarityFrame2(predMaps(:,:,indFr(ifr)), gazeParam.gazeData{frames(indFr(ifr))}, gazeParam.gazeData(frames(indFr([1:indFr(ifr)-1, indFr(ifr)+1:end]))), measures, ...
                'self', ...
                struct('method', 'center', 'cov', [(n/16)^2, 0; 0, (n/16)^2]), ...
                struct('method', 'saliency_GBVS', 'map', fr.saliency), ...
                struct('method', 'saliency_PQFT', 'map', fr.saliencyPqft), ...
                struct('method', 'saliency_Hou', 'map', fr.saliencyHou));
            if (visVideo && verNum >= 2012)
                outfr = renderSideBySide(fr.image, outMaps, colors, cmap, sim{i}(:,:,ifr));
                writeVideo(vw, outfr);
            end
        end
    catch me
        if (visVideo && verNum >= 2012)
            close(vw);
        end
        rethrow(me);
    end
    
    if (visVideo && verNum >= 2012)
        close(vw);
    end

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
    if (size(meanChiSq, 1) == 1), meanChiSq = [meanChiSq; zeros(1, length(methods))]; end;
    
    figure, bar(meanChiSq);
    imLabel(lbl, 'bottom', -90, {'FontSize',8, 'Interpreter', 'None'});
    ylabel(measures{im});
    title(sprintf('Mean %s', mat2str(meanMeas, 2)));
    legend(methods);
    
    print('-dpng', fullfile(visRoot, sprintf('overall_%s.png', measures{im})));
end
