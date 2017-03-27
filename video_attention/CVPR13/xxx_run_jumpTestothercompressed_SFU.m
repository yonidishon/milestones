% run_jumpTest
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VERSIONS
% v6 - trained for CVPR, using cvpr13_v5_3 model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% settings
addpath(fullfile('C:\Users\ydishon\Documents\milestones\video_attention\CVPR13','xxx_my_additions'));
settings()
modelfeaturesaveloc = '\\cgm47\D\Dima_Analysis_Milestones\ModelsFeatures'; %TODO
diemDataRoot = '\\cgm47\D\Competition_Dataset\SFU';

VERSION = {'MAM','MCSDM','MSM-SM','PIM-MCS','PIM-ZEN','PMES','PNSP-CS'}; % TODO
for k=1:length(VERSION)
OBDLsavedfold = '\\cgm47\d\Competition_Dataset\SFU\ResultsCompressedAlgo';
uncVideoRoot = fullfile(diemDataRoot, 'avi');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
visRoot = fullfileCreate(modelfeaturesaveloc,'PredictionsNO_SEM_SFU', VERSION{k}); %TODO

jumpType = 'all'; % 'cut' or 'gaze_jump' or 'random' or 'all'
sourceType = 'rect';
% measures = {'chisq', 'auc', 'cc', 'nss'};
measures = {'chisq', 'auc','nss'};
methods = {'proposed', 'self'};

% cache settings
cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.gazeRoot = '\\cgm47\D\Competition_Dataset\SFU\gaze';
cache.renew = false; % use in case the preprocessing mechanism updated

% gaze settings
gazeParam.pointSigma = 10;

%testing settings
testIdx = 1:12;
testSubset = 1:length(testIdx);
jumpFromType = 'prev-int'; % 'center', 'gaze', 'prev-cand', 'prev-int'
visVideo = false;
candScale = 2;

% visualization
cmap = jet(256);
colors = [1 0 0;
    0 1 0;
    0 0 1;
    1 1 0;
    1 0 1;
    0 1 1];

%% prepare
%videos = videoListLoad(diemDataRoot, 'DIEM');
videos = extractfield(dir(fullfile(diemDataRoot,'DATA')),'name');
videos = videos(3:end);
nv = length(videos);

% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

nt = length(testSubset);
sim = cell(nt, 1);

vers = version('-release');
verNum = str2double(vers(1:4));

if (~exist(visRoot, 'dir'))
    mkdir(visRoot);
end

%% test
for i = 1:length(videos) %TODO
    iv = testIdx(testSubset(i));
    fprintf('Time is:: %s, Processing %s... ',datestr(datetime('now')), videos{iv}); tic;
    
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
    [jumpFrames, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType, 50, 1, videoLen - 1);
    nc = length(jumpFrames);
    
    % load gaze data
    s = load(fullfile(cache.gazeRoot, sprintf('%s.mat', videos{iv})));
    gazeData = s.data;
    clear s;
    gazeData.pointSigma = gazeParam.pointSigma;
    gazeParam.gazeData = gazeData.points;
    videoLen = min(videoLen, length(gazeData.points));  
    resFile = fullfile(visRoot, sprintf('%s.mat', videos{iv}));
    predMaps = load(fullfile(OBDLsavedfold,videos{iv},sprintf('result_%s_H264_QP30.mat',VERSION{k})));
    predMaps=predMaps.S;
    predMaps = imresize(predMaps,[144,174]);
    % compare
    frames = jumpFrames + after;
    indFr = find(frames <= videoLen);
            
    sim{i} = zeros(length(methods), length(measures), length(indFr));  
    
    for ifr = 1:length(indFr)
        gazeData.index = frames(indFr(ifr));
        [sim{i}(:,:,ifr), outMaps] = similarityFrame3(predMaps(:,:,indFr(ifr)), gazeData, measures, ...
            'self');
       
        if mod(ifr,100) == 0
            fprintf('Time is:: %s,Frame Evaluated: %d/%d\n',datestr(datetime('now')),ifr,length(indFr))
        end
    end

    vid_sim = sim{i};
    save(fullfile(visRoot, sprintf('%s_similarity.mat',videos{iv})),'vid_sim');
    clear vid_sim;
    fprintf('%f sec\n', toc);
end

save(fullfile(visRoot, '00_similarity.mat'), 'sim', 'measures', 'methods', 'testIdx', 'testSubset');
fprintf('Finished to compute similarity on %s\n',datestr(datetime('now')));

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
    
    %print('-dpng', fullfile(visRoot, sprintf('overall_%s.png', measures{im})));
end

% histogram
visCompareMethods(sim, methods, measures, videos, testIdx(testSubset), 'boxplot');
fprintf('Finished to Evaluate on %s\n',datestr(datetime('now')));
end