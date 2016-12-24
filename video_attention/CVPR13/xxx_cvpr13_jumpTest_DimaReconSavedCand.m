clear vars;close all;clc;
%% settings
% First guess:
global gdrive
global dropbox
saveloc = '\\cgm47\D\Dima_Analysis_Milestones\DimaResultsSavedCandOldChi';

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


    cache.featureRoot = fullfile(cache.root, '00_features_v6');
    visRoot = fullfile(saveloc);
    for i = 1:nt
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
        % best results of Dmitry folder 
        
        resFile = fullfile('\\cgm10\Users\ydishon\Documents\Video_Saliency\DimaResults\jump_test_v6_orig', sprintf('%s.mat', videos{iv}));
        fprintf('(loading)...\n');
        s = load(resFile);
        frames = s.frames;
        indFr = s.indFr;
        cands = s.cands;
        predMaps = s.predMaps;
        clear s;
%         predMaps = zeros(m, n, nc); % dense maps     
%         for ic = 1:nc
%             predMaps(:,:,ic) = candidate2map(cands{ic}, [n, m], candScale);
%         end
        
        % compare
        frames = jumpFrames + after;
        indFr = find(frames <= videoLen);
        
        % save
    %    save(fullfile(visRoot, sprintf('%s.mat', videos{iv})), 'frames', 'indFr', 'cands', 'predMaps');
     
        
        % similarity
        sim{i} = zeros(length(methods), length(measures), length(indFr));
        fprintf('Time is :: %s , Calculating Similarity...\n',datestr(datetime));
        
        for ifr = 1:length(indFr)
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