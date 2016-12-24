clear vars;close all;clc;
%% settings
% First guess:
global gdrive
global dropbox
saveloc = '\\cgm47\D\Dima_Analysis_Milestones\DimaResults';

diemDataRoot = '\\cgm47\D\Dimtry_orig\DIEM';

addpath(fullfile(pwd,'xxx_my_additions'));
gdrive = '\\cgm10\Users\ydishon\Documents\Video_Saliency\Dimarudoy_saliency\GDrive';
dropbox = '\\cgm10\Users\ydishon\Documents\Video_Saliency\Dimarudoy_saliency\Dropbox';
addpath(genpath(fullfile(gdrive, 'Software', 'dollar_261')));
addpath(fullfile(gdrive, 'Software', 'OpticalFlow')); % optical flow
addpath(fullfile(gdrive, 'Software', 'OpticalFlow\mex'));
addpath(genpath(fullfile(gdrive, 'Software', 'Saliency', 'gbvs'))); % GBVS saliency
addpath(genpath(fullfile(gdrive, 'Software', 'Saliency', 'Hou2008'))); % Houw saliency
addpath(genpath(fullfile(gdrive,'Software','Saliency','qtfm')));%PQFT
addpath(genpath(fullfile(dropbox,'Matlab','video_attention','compare','PQFT_2'))); % pqft

addpath(genpath(fullfile(dropbox,'Software','face_detector_adobe'))); % face detector
addpath(genpath(fullfile(dropbox,'Software','poselets','code')));% poselets

addpath(genpath(fullfile(gdrive,'Software','MeanShift')));% meanShift
addpath(genpath(fullfile(gdrive,'Software','misc')));% melliecious

addpath(genpath(fullfile(gdrive,'Software','randomforest-matlab'))); %Random forest

addpath(genpath(fullfile(dropbox,'Matlab','video_attention','compare','BorjiMetrics')));

uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = '\\cgm47\D\DIEM\gaze';

modelFile = fullfile('\\cgm47\D\DimaReleaseCode_CGMwebsite\data', '00_trained_model_cvpr13_v5_3.mat'); % CVPR13

jumpType = 'all'; % 'cut' or 'gaze_jump' or 'random' or 'all'
sourceType = 'cand'; %'rect';
measures = {'chisq', 'auc', 'nss'};
%measures = {'chisq', 'auc'};
methods = {'Rudoy et al.', 'self'};%, 'center', 'GBVS', 'RGBD', 'PQFT', 'Hou'};
DataRoot = '\\cgm47\D\Dimtry_orig\DIEM';
% % cache settings
cache.root = fullfile(DataRoot, 'cache');
cache.frameRoot = fullfile(DataRoot, 'cache');
cache.featureRoot = fullfile(cache.root, '00_features_v6');
cache.gazeRoot = fullfile(cache.root, '00_gaze');
% cache.renew = true; % use in case the preprocessing mechanism updated
% cache.renewFeatures = true; % use in case the feature extraction is updated
% cache.renewJumps = true; % recalculate the final result

% gaze settings
gazeParam.pointSigma = 10;

% training and testing settings
testIdx = [6,8,10,11,12,14,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % used by Borji
testIdx = [testIdx(6),testIdx(1:5),testIdx(7:end)]; % just re-arranging so it will be fast
% testSubset = 11:length(testIdx);
% testSubset = 9;
jumpFromType = 'prev-int'; % 'center', 'gaze', 'prev-cand', 'prev-int'
candScale = 2;

% visualization
cmap = jet(256);
colors = [1 0 0;
    0 1 0;
    0 0 1;
    1 1 0;
    1 0 1;
    0.5 0.5 0.5;
    0 1 1];

%% prepare
videos = videoListLoad(DataRoot, 'DIEM');
nv = length(videos);

%testIdx = 2:2:nv;
testSubset = 1:length(testIdx);


% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

% load model
s = load(modelFile);
rf = s.rf;
options = s.options;
options.useLabel = false; % no need in label while testing
clear s;

vers = version('-release');
verNum = str2double(vers(1:4));

nt = length(testSubset);
%% test
ff = {'00_features_v6','00_features_v5','00_features_v7'};
for kk=1:1
    cache.featureRoot = fullfile(cache.root, ff{kk});
    visRoot = fullfile(saveloc,ff{kk});
    for i = 4:nt
        iv = testIdx(testSubset(i));
        fprintf('%s :: Processing %s... \n', datestr(datetime), videos{iv});tic;
        
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
        [jumpFrames, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType, 50, 30, videoLen - 30);
        nc = length(jumpFrames);
        
        % load gaze data
        s = load(fullfile(cache.gazeRoot, sprintf('%s.mat', videos{iv})));
        gazeData = s.data;
        clear s;
        gazeParam.gazeData = gazeData.points;
        videoLen = min(videoLen, length(gazeData.points));
        
        % load resut file if exist:
        resFile = fullfile(visRoot, sprintf('%s.mat', videos{iv}));
        if (exist(resFile, 'file')) % load from cache
            fprintf('(loading)... ');
            s = load(resFile);
            frames = s.frames;
            indFr = s.indFr;
            cands = s.cands;
            predMaps = s.predMaps;
            clear s;
        else % calculateelse % calculate
            cands = cell(nc, 1); % candidates per frame
            predMaps = zeros(m, n, nc); % dense maps
            
            for ic = 1:nc
                fr = xxx_preprocessFrames_DimaRecon(vr,  jumpFrames(ic)+before, gbvsParam, ofParam, poseletModel,cache);
                if ((jumpFrames(ic) + before >= 3) && (jumpFrames(ic) + after <= videoLen))
                    % source frame
                    if (strcmp(jumpFromType, 'center')) % jump from center
                        srcCands = {struct('point', [n/2, m/2], 'score', 1, 'type', 1, 'candCov', [(m/8)^2, 0; 0, (m/8)^2])}; % dummy source at center
                    elseif (strcmp(jumpFromType, 'prev-cand')) % jump from previous candidate set
                        if (ic == 1 || isempty(cands{ic-1})) % first or empty previous
                            srcCands = {struct('point', [n/2, m/2], 'score', 1, 'type', 1, 'candCov', [(m/8)^2, 0; 0, (m/8)^2])}; % dummy source at center
                        else
                            srcCands = cands{ic-1};
                        end
                    elseif (strcmp(jumpFromType, 'prev-int')) % jump from previous integrated map
                        if (ic == 1 || isempty(cands{ic-1})) % first or empty previous
                            srcCands = {struct('point', [n/2, m/2], 'score', 1, 'type', 1, 'candCov', [(m/8)^2, 0; 0, (m/8)^2])}; % dummy source at center
                        else
                            srcCands = xxx_sourceCandidates_DimaRecon(fr, predMaps(:,:,ic-1), options, sourceType);
                        end
                    else
                        error('Unsupported jump from type: %s', jumpFromType);
                    end
                    dstCands = xxx_jumpPerform6_DimaRecon(srcCands, jumpFrames(ic)+before, jumpFrames(ic)+after, param, options, gbvsParam, ofParam, poseletModel, rf, cache);
                    %ic
                    predMaps(:,:,ic) = candidate2map(dstCands, [n, m], candScale);
                    cands{ic} = dstCands;
                end
            end
            
            % compare
            frames = jumpFrames + after;
            indFr = find(frames <= videoLen);
            
            % save
            save(fullfile(visRoot, sprintf('%s.mat', videos{iv})), 'frames', 'indFr', 'cands', 'predMaps');
        end % if saved results
        
        
        % similarity
        sim{i} = zeros(length(methods), length(measures), length(indFr));
        fprintf('Time is :: %s , Calculating Similarity...\n',datestr(datetime));
        
        for ifr = 1:length(indFr)
            fr = xxx_preprocessFrames_DimaRecon(param.videoReader, frames(indFr(ifr)), gbvsParam, ofParam, poseletModel, cache);
            gazeData.index = frames(indFr(ifr));
            [sim{i}(:,:,ifr), outMaps] = similarityFrame3(predMaps(:,:,indFr(ifr)), gazeData, measures, ...
                'self');
        end
        
        fprintf('Time is :: %s , It Took :: %f sec\n',datestr(datetime), toc);
    end
    save(fullfile(visRoot, '00_similarity.mat'), 'sim', 'measures', 'methods', 'testIdx', 'testSubset');
    
    %% visualize
    fprintf('Time is:: %s ,  Analyzing Results....\n',datestr(datetime));
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
    
    % histogram
    visCompareMethods(sim, methods, measures, videos, testIdx(testSubset), 'boxplot', visRoot);
    fprintf('Time is :: %s , Finished! \n',datestr(datetime), toc);
end