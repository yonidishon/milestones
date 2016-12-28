% cvpr13_validateCands
%clear all;close all;clc;
%% settings
% First guess:
addpath(fullfile('C:\Users\ydishon\Documents\milestones\video_attention\CVPR13','xxx_my_additions'));
settings()
saveloc = '\\cgm47\D\Dima_Analysis_Milestones\Candidates\PrefectPredExp';
diemDataRoot = '\\cgm47\D\DIEM';

uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');

videoIdx = [6,8,10,11,12,14,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % used by Borji;
jumpType = 'all';
candScale = 2;
visVideo = false;
measures = {'chisq', 'auc','nss'};
% gaze settings
gazeParam.pointSigma = 10;
gazeRoot = fullfile(diemDataRoot, 'cache', '00_gaze');
%% loading
nv = length(videoIdx);
videos = videoListLoad(diemDataRoot);

uncVideoRoot = fullfile(diemDataRoot, 'video_unc');

[gbvsParam, ofParam, poseletModel] = configureDetectors();
%Extracting only directories of models that I tested on
dataloc = '\\cgm47\d\Dima_Analysis_Milestones\ModelsFeatures\PredictionsNO_SEM';
files=dir(dataloc);
isdirbool = cell2mat(extractfield(files,'isdir'));
filenames = extractfield(files,'name');
dirnames = filenames(isdirbool);
dirnames = dirnames(~ismember(dirnames,{'.','..'}));
for kk=1:length(dirnames);
    visRoot = fullfile(saveloc,dirnames{kk});
    if ~exist('visRoot','dir')
        mkdir(visRoot);
    end
    sim = cell(nv, 1);
    for i = 1:nv % all movies
        iv = videoIdx(i);
        videoName = videos{iv};
        
        fprintf('%s :: Processing %s ... \n', datestr(datetime('now')), videoName); tic;
        % compare
        % load jump frames
        vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
        videoLen =  vr.numberOfFrames;
        [jumpFrames, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType, 50, 30,  videoLen - 30);
        frames = jumpFrames + after;
        indFr = find(frames <= videoLen);
        % load candidates
        data = load(fullfile(dataloc,dirnames{kk},sprintf('%s.mat',videoName)));
        cands = data.cands;
        % load gaze data
        ss = load(fullfile(gazeRoot, sprintf('%s.mat', videoName)));
        gazeData = ss.data;
        h = ss.data.height;
        w = ss.data.width;
        clear ss;
        predmaps = zeros(h,w,length(indFr));
        for ifr = 1:length(indFr) % all frames
            gazeData.index = frames(indFr(ifr));
            predmaps(:,:,ifr) = xxx_candGazePrefectPred(gazeData.points{gazeData.index}, cands{ifr},h,w);
            [sim{i}(:,:,ifr), ~] = similarityFrame3(predmaps(:,:,indFr(ifr)), gazeData, measures, ...
                'self');
        end
        vid_sim = sim{i};
        save(fullfile(visRoot, sprintf('%s.mat', videos{iv})), 'frames', 'indFr', 'cands', 'predmaps','vid_sim');
        
        
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