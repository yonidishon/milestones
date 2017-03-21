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

VERSION = 'NoCachedPartial'; % TODO

uncVideoRoot = fullfile(diemDataRoot, 'avi');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
visRoot = fullfileCreate(modelfeaturesaveloc,'PredictionsNO_SEM_SFU', VERSION); %TODO
% modelFile = fullfile(uncVideoRoot, '00_trained_model_validation_v5_3.mat'); % validation
modelFile = fullfile(modelfeaturesaveloc, sprintf('00_trained_model_cached_v5_3_no_sem_%s.mat',VERSION)); %TODO

jumpType = 'all'; % 'cut' or 'gaze_jump' or 'random' or 'all'
sourceType = 'rect';
% measures = {'chisq', 'auc', 'cc', 'nss'};
measures = {'chisq','auc','nss'};
methods = {'proposed', 'self'};

% cache settings
cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfileCreate(modelfeaturesaveloc, sprintf('00_features_v6%s',VERSION));
cache.gazeRoot = '\\cgm47\D\Competition_Dataset\SFU\gaze';
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = true; %TODO % use in case the feature extraction is updated
cache.renewJumps = false; % recalculate the final result

% gaze settings
gazeParam.pointSigma = 10;

%testing settings
testIdx = 1:12;
testSubset = 1:length(testIdx);
%testSubset = [6,9,12,13]; % see cvpr13_jumpTestPERcUE.M
% testSubset = 11:length(testIdx);
% testSubset = 9;
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
videos = extractfield(dir(fullfile(diemDataRoot,'DATA')),'name');
videos = videos(3:end);
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
for i = 1:length(videos) %TODO
    iv = testIdx(testSubset(i));
    fprintf('Processing %s... ', videos{iv}); tic;
%     
%     % prepare video
%     if (isunix) % use matlab video reader on Unix
%         vr = VideoReaderMatlab(fullfile(uncVideoRoot, sprintf('%s.mat', videos{iv})));
%     else
%         if (verNum < 2011)
%             vr = mmreader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
%         else
%             vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
%         end
%     end
%     m = vr.Height;
%     n = vr.Width;
%     videoLen = vr.numberOfFrames;
%     param = struct('videoReader', vr);
%     
%     % load jump frames
%     [jumpFrames, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType, 50, 30, videoLen - 30);
%     nc = length(jumpFrames);
%     
    % load gaze data
    s = load(fullfile(cache.gazeRoot, sprintf('%s.mat', videos{iv})));
    gazeData = s.data;
    clear s;
    gazeData.pointSigma = gazeParam.pointSigma;
    gazeParam.gazeData = gazeData.points;
% 
%     videoLen = min(videoLen, length(gazeData.points));
%     
%     resFile = fullfile(visRoot, sprintf('%s.mat', videos{iv}));
% %     if (~cache.renewJumps && exist(resFile, 'file')) % load from cache
% %         fprintf('(loading)... ');
% %         s = load(resFile);
% %         frames = s.frames;
% %         indFr = s.indFr;
% %         cands = s.cands;
% %         predMaps = s.predMaps;
% %         clear s;
%         
% %    else % calculate
%         cands = cell(nc, 1); % candidates per frame
%         predMaps = zeros(m, n, nc); % dense maps
%         
%         for ic = 1:nc
%             if ((jumpFrames(ic) + before >= 3) && (jumpFrames(ic) + after <= videoLen))
%                 % source frame
%                 if (strcmp(jumpFromType, 'center')) % jump from center
%                     srcCands = {struct('point', [n/2, m/2], 'score', 1, 'type', 1, 'candCov', [(m/8)^2, 0; 0, (m/8)^2])}; % dummy source at center
%                 elseif (strcmp(jumpFromType, 'gaze')) % jump from gaze map
%                     srcGazeMap = points2GaussMap(gazeParam.gazeData{jumpFrames(ic)+before}', ones(1, size(gazeParam.gazeData{jumpFrames(ic)+before}, 1)), 0, [n, m], gazeParam.pointSigma);
%                     [srcCands, ~, ~, ~] = sourceCandidates([], srcGazeMap, options, sourceType);
%                 elseif (strcmp(jumpFromType, 'prev-cand')) % jump from previous candidate set
%                     if (ic == 1 || isempty(cands{ic-1})) % first or empty previous
%                         srcCands = {struct('point', [n/2, m/2], 'score', 1, 'type', 1, 'candCov', [(m/8)^2, 0; 0, (m/8)^2])}; % dummy source at center
%                     else
%                         srcCands = cands{ic-1};
%                     end
%                 elseif (strcmp(jumpFromType, 'prev-int')) % jump from previous integrated map
%                     if (ic == 1 || isempty(cands{ic-1})) % first or empty previous
%                         srcCands = {struct('point', [n/2, m/2], 'score', 1, 'type', 1, 'candCov', [(m/8)^2, 0; 0, (m/8)^2])}; % dummy source at center
%                     else
%                         srcCands = xxx_sourceCandidatesForTrainDimtry([], predMaps(:,:,ic-1), options, sourceType);
%                     end
%                 else
%                     error('Unsupported jump from type: %s', jumpFromType);
%                 end
%                 
%                 dstCands = xxx_jumpPerform6Partial(srcCands, jumpFrames(ic)+before, jumpFrames(ic)+after, param, options, gbvsParam, ofParam, poseletModel, rf,cache); %TODO
%                 predMaps(:,:,ic) = candidate2map(dstCands, [n, m], candScale);
%                 cands{ic} = dstCands;
%             end
%         end
%         
%         % compare
%         frames = jumpFrames + after;
%         indFr = find(frames <= videoLen);
%         
%         % save
%         save(fullfile(visRoot, sprintf('%s.mat', videos{iv})), 'frames', 'indFr', 'cands', 'predMaps');
%     
%  %   end
    
    % visualize
    videoFile = fullfile(visRoot, sprintf('%s.avi', videos{iv}));
    saveVideo = visVideo && (~exist(videoFile, 'file'));
        ss = load(fullfile(visRoot,sprintf('%s.mat', videos{iv})));
    predMaps = ss.predMaps;
    indFr = ss.indFr;
    frames = ss.frames;
    clear ss;    
    sim{i} = zeros(length(methods), length(measures), length(indFr));
%     if (saveVideo && verNum >= 2012)
%         vw = VideoWriter(videoFile, 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
%         open(vw);
%     end

    try
        for ifr = 1:length(indFr)
            % fr = xxx_preprocessFramesPartial(param.videoReader, frames(indFr(ifr)), gbvsParam, ofParam, poseletModel, cache);
            gazeData.index = frames(indFr(ifr));
            [sim{i}(:,:,ifr), outMaps] = similarityFrame3(predMaps(:,:,indFr(ifr)), gazeData, measures, ...
                'self');
            if (saveVideo && verNum >= 2012)
                outfr = renderSideBySide(fr.image, outMaps, colors, cmap, sim{i}(:,:,ifr));
                writeVideo(vw, outfr);
            end
        end
    catch me
        if (saveVideo && verNum >= 2012)
            close(vw);
        end
        rethrow(me);
    end
    
    if (saveVideo && verNum >= 2012)
        close(vw);
    end

    fprintf('%f sec\n', toc);
end

save(fullfile(visRoot, '00_similarity.mat'), 'sim', 'measures', 'methods', 'testIdx', 'testSubset');
% save(fullfile(visRoot, '00_similarity_1_2.mat'), 'sim', 'measures', 'methods', 'testIdx', 'testSubset');
% fprintf('Finished to compute similarity of 1-2 on %s\n',datestr(datetime('now')));
% secondsim = load(fullfile(visRoot,'00_similarity.mat'));
% secondsim = secondsim.sim;
% for ij=3:20
%     sim{ij}= secondsim{ij};
% end
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
