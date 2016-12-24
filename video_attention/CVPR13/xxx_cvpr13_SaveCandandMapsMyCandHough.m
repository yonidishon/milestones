% cvpr13_validateCands
clear vars;close all;clc;
%% settings
global gdrive 
global dropbox
pcaloc = '\\cgm47\D\head_pose_estimation\Predictions\2016_02_27_Post_MRF2';
myCandssaveLoc = '\\cgm47\D\Dima_Analysis_Milestones\Candidates\MyCandHough';

cand_test_v6orig = '\\cgm10\Users\ydishon\Documents\Video_Saliency\DimaResults\jump_test_v6_orig';

dimaCandFold = {cand_test_v6orig};

diemDataRoot = '\\cgm47\D\DIEM';
addpath(fullfile('\\cgm10\Users\ydishon\Documents\Video_Saliency\Dimarudoy_saliency\Dropbox\Matlab\video_attention\CVPR13','xxx_my_additions'));
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

outRoot = fullfileCreate(diemDataRoot, 'cvpr13');
modelFile = fullfile(uncVideoRoot, '00_trained_model_cvpr13_v5_3.mat'); % CVPR13

videoIdx = [14,6,8,10,11,12,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % used by Borji;
candScale = 2;
visVideo = false;

% cache settings
cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfileCreate(cache.root, '00_features_v6');
cache.gazeRoot = fullfileCreate(cache.root, '00_gaze');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = false; % use in case the feature extraction is updated
cache.renewJumps = false; % recalculate the final result

% gaze settings
gazeParam.pointSigma = 10;

% visualization
cmap = jet(256);
colors = [1 0 0;
    0 1 0;
    0 0 1;
    1 1 0;
    1 0 1;
    0 1 1];


%% loading
visRoot = dimaCandFold{1};
nv = length(videoIdx);
videos = videoListLoad(diemDataRoot);
[gbvsParam, ofParam, poseletModel] = configureDetectors();
ss = load(modelFile);
options = ss.options;
candScale = 2;
%% run
% all videos in testset
for i = 1:nv
    iv = videoIdx(i);
    videoName = videos{iv};

    fprintf('Processing %s... ', videoName); tic;

    s = load(fullfile(visRoot, sprintf('%s.mat', videoName)));
    [m, n, nfr] = size(s.predMaps);
    
    vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
    
    mycands = cell(length(s.indFr),1);
    pcam = load(fullfile(pcaloc,[videoName,'.mat']));
    pcam = pcam.predMaps;
    for ifr = 1:length(s.indFr)
        % GET VISUAL FRAME
        mycands{ifr} = xxx_jumpCandidates3simpleMyCand(im2double(pcam(:,:,ifr)), options); 
    end
    
    % save the data to the corresponding locations:
    frames = s.frames;
    save(fullfile(myCandssaveLoc,sprintf('%s.mat',videos{iv})),'mycands','frames');
    fprintf('%s took %f sec\n',videos{iv},toc); 
end


