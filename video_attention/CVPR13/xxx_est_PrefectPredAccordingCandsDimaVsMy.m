% cvpr13_validateCands
%clear all;close all;clc;
%% settings
% First guess:
global gdrive
global dropbox
saveloc = '\\cgm47\D\Dima_Analysis_Milestones\Candidates\PrefectPredExp';
diemDataRoot = '\\cgm47\D\DIEM';
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

uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');

videoIdx = [6,8,10,11,12,14,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % used by Borji;

candScale = 2;
visVideo = false;
measures = {'chisq', 'auc','nss'};
% gaze settings
gazeParam.pointSigma = 10;

%% loading
nv = length(videoIdx);
videos = videoListLoad(diemDataRoot);


[gbvsParam, ofParam, poseletModel] = configureDetectors();
%Extracting only directories of models that I tested on
dataloc = '\\cgm47\d\Dima_Analysis_Milestones\ModelsFeatures\PredictionsNO_SEM';
files=dir();
isdirbool = cell2mat(extractfield(files,'isdir'));
filenames = extractfield(files,'name');
dirnames = filenames(isdirbool);
dirnames = dirnames(~ismember(dirnames,{'.','..'}));
for kk=1:length(dirnames);
    visRoot = fullfile(saveloc,dirnames{kk});
    sim = cell(nv, 1);
    for i = 1:nv % all movies
        iv = videoIdx(i);
        videoName = videos{iv};
        
        fprintf('% :: Processing %s... ', datestr(datetime), videoName); tic;
        % compare
        % load jump frames
        [jumpFrames, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType, 50, 30, videoLen - 30);
        frames = jumpFrames + after;
        indFr = find(frames <= videoLen);
        % load candidates
        data = load(fullfile(dataloc,dirnames{kk},sprintf('%s.mat',videoName)));
        cands = data.cands;
        % load gaze data
        ss = load(fullfile(cache.gazeRoot, sprintf('%s.mat', videoName)));
        gazeData = ss.data;
        h = ss.height;
        w = ss.width;
        clear ss;
        predmaps = zeros(h,w,nfr);
        for ifr = 1:length(indFr) % all frames
            gazeData.index = frames(indFr(ifr));
            predmaps(:,:,ifr) = xxx_candGazePrefectPred(gazeData.points{gazeData.index}, cands{gazeData.index},h,w);
            [sim{i}(:,:,ifr), ~] = similarityFrame3(predMaps(:,:,indFr(ifr)), gazeData, measures, ...
                'self');
        end
        vid_sim = sim{i};
        save(fullfile(visRoot, sprintf('%s.mat', videos{iv})), 'frames', 'indFr', 'cands', 'predMaps','vid_sim');
        
        
        %fprintf('%f sec\n', toc);
    end
    fprintf('%s Finished data collection Directory %s:  %d/%d \n',datestr(datetime),dirnames{kk},kk,length(dirnames));
    
    %% results
    save(fullfile(visRoot, '00_similarity.mat'), 'sim', 'measures', 'methods', 'testIdx', 'testSubset');
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
    fprintf('Finished to Evaluate on %s on Directory %s\n',datestr(datetime('now')),dirnames{kk});
end
restoredefaultpath;