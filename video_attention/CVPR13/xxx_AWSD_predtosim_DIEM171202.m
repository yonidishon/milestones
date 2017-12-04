% run_jumpTest
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VERSIONS
% v6 - trained for CVPR, using cvpr13_v5_3 model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% settings
addpath(fullfile('C:\Users\ydishon\Documents\milestones\video_attention\CVPR13','xxx_my_additions'));
settings()
modelfeaturesaveloc = '\\cgm47\D\AWSD_P4\xDIEM'; %TODO
diemDataRoot = '\\cgm47\D\DIEM';
VERSION = 'AWSD_P4171202'; % TODO

uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
visRoot = fullfileCreate(modelfeaturesaveloc, VERSION); %TODO
% modelFile = fullfile(uncVideoRoot, '00_trained_model_validation_v5_3.mat'); % validation

jumpType = 'all'; % 'cut' or 'gaze_jump' or 'random' or 'all'
sourceType = 'rect';
% measures = {'chisq', 'auc', 'cc', 'nss'};
measures = {'chisq', 'auc','nss'};
methods = {'proposed', 'self'};

% cache settings
cache.root = fullfile(diemDataRoot, 'cache');
cache.gazeRoot = fullfile(cache.root,'00_gaze');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = true;%TODO % use in case the feature extraction is updated
cache.renewJumps = false; % recalculate the final result

% gaze settings
gazeParam.pointSigma = 10;

% training and testing settings
testIdx = [6,8,10,11,12,14,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % used by Borji

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
% videos = videoListLoad(diemDataRoot, 'DIEM');
videos = videoListLoad(diemDataRoot, 'DIEM');
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
    fprintf('Time is:: %s, Processing %s... \n',datestr(datetime('now')), videos{iv}); tic;
    
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
    
%     % load jump frames
    [jumpFrames, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType, 50, 30, videoLen - 30);
    nc = length(jumpFrames);
    
    % load gaze data
    s = load(fullfile(cache.gazeRoot, sprintf('%s.mat', videos{iv})));
    gazeData = s.data;
    clear s;
    gazeData.pointSigma = gazeParam.pointSigma;
    gazeParam.gazeData = gazeData.points;

    videoLen = min(videoLen, length(gazeData.points));
    
       
    %predMaps = zeros(m, n, nc); % dense maps
    predMaps = load(fullfile(modelfeaturesaveloc,[videos{iv},'.mat']),'predMaps');
    predMaps = im2double(predMaps.predMaps);

%         for ic = 1:nc
%             if ((jumpFrames(ic) + before >= 3) && (jumpFrames(ic) + after <= videoLen))
%                
%             end
%             if mod(ic,100) == 0
%                  fprintf('Time is:: %s,Frame Processed: %d/%d\n',datestr(datetime('now')),ic,nc)
%             end
%         end
%         
        % compare
        frames = jumpFrames + after;
        indFr = find(frames <= videoLen);
        
    
    % visualize
%     videoFile = fullfile(visRoot, sprintf('%s.avi', videos{iv}));
%     saveVideo = visVideo && (~exist(videoFile, 'file'));
        
    sim{i} = zeros(length(methods), length(measures), length(indFr));
%     if (saveVideo && verNum >= 2012)
%         vw = VideoWriter(videoFile, 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
%         open(vw);
%     end
    
    for ifr = 1:length(indFr)
        gazeData.index = frames(indFr(ifr));
        [sim{i}(:,:,ifr), outMaps] = similarityFrame3(predMaps(:,:,indFr(ifr)), gazeData, measures, ...
            'self');
%         if (saveVideo && verNum >= 2012)
%             outfr = renderSideBySide(fr.image, outMaps, colors, cmap, sim{i}(:,:,ifr));
%             writeVideo(vw, outfr);
%         end
        if mod(ifr,100) == 0
            fprintf('Time is:: %s,Frame Evaluated: %d/%d\n',datestr(datetime('now')),ifr,length(indFr))
        end
    end
    % For case of faliure during the run
    vid_sim = sim{i};
    save(fullfile(visRoot, sprintf('%s_similarity.mat',videos{iv})),'vid_sim');
    clear vid_sim;
    fprintf('%f sec\n', toc);
end

save(fullfile(visRoot, '00_similarity.mat'), 'sim', 'measures', 'methods', 'testIdx', 'testSubset');
fprintf('Finished to compute similarity on %s\n',datestr(datetime('now')));
% % FOR THE EVENT THAT SOMETHING IS STOPPED IN THE MIDDLE
sim = cell(nt, 1);
for ii = 1:nt
    sim{ii} = load(fullfile(visRoot,sprintf('%s_similarity.mat',videos{testIdx(ii)})));
    sim{ii} = sim{ii}.vid_sim;
end
%% visualize
fprintf('Time is: %s, Statistics computation!\n',datestr(datetime('now')));
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