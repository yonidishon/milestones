clear vars;close all;clc;
%% settings
% First guess:
global gdrive 
global dropbox
saveloc = 'D:\Dima_Analysis_Milestones\Candidates';
visRoot = fullfile(saveloc,'Dima_reconstruction');
cand_test_v6 = 'D:\DIEM\cvpr13\jump_test_v6';
%cand_test_v7 = 'D:\DIEM\cvpr13\jump_test_v7'; % This on seems to be the
%GT candidates
cand_test_v6orig = 'C:\Users\ydishon\Documents\Video_Saliency\DimaResults\jump_test_v6_orig';
cand_test_v61 = 'C:\Users\ydishon\Documents\Video_Saliency\DimaResults\jump_test_v6';

%dimaCandFold = {cand_test_v6, cand_test_v6orig,cand_test_v61};
dimaCandFold = {cand_test_v6orig};

diemDataRoot = 'D:\DIEM';
addpath(fullfile(pwd,'xxx_my_additions'));
gdrive = 'C:\Users\ydishon\Documents\Video_Saliency\Dimarudoy_saliency\GDrive';
dropbox = 'C:\Users\ydishon\Documents\Video_Saliency\Dimarudoy_saliency\Dropbox';
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

jumpType = 'all'; % 'cut' or 'gaze_jump' or 'random' or 'all'
sourceType = 'cand'; %'rect';
% measures = {'chisq', 'auc', 'cc', 'nss'};
%measures = {'chisq', 'auc'};
%methods = {'proposed', 'self', 'center', 'GBVS', 'RGBD', 'PQFT', 'Hou'};

% % cache settings
% cache.root = fullfile(DataRoot, 'cache');
% cache.frameRoot = fullfile(DataRoot, 'cache');
% cache.featureRoot = fullfileCreate(cache.root, '00_features_v6');
% cache.gazeRoot = fullfileCreate(cache.root, '00_gaze');
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
videos = videoListLoad(diemDataRoot, 'DIEM');
nv = length(videos);

%testIdx = 2:2:nv;
testSubset = 1:length(testIdx);


% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

% load model
s = load(modelFile);
rf = s.rf;
options = s.options;
options= rmfield(options,'topCandsNum');
options.useLabel = false; % no need in label while testing
clear s;

vers = version('-release');
verNum = str2double(vers(1:4));

if (~exist(visRoot, 'dir'))
    mkdir(visRoot);
end

nt = length(testSubset);
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
    
    
    % load jump frames
    [jumpFrames, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType, 50, 30, videoLen - 30);
    nc = length(jumpFrames);
      
    cands = cell(nc, 1); % candidates per frame
    predMaps = zeros(m, n, nc); % dense maps
    
    for ic = 1:1%nc
        %fr = preprocessFrames(vr,  jumpFrames(ic)+before, gbvsParam, ofParam, poseletModel);
        savloc = 'c:\Users\ydishon\Documents\tmp\fr.mat';
        fr = importdata(savloc,'fr');
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
                    srcCands = sourceCandidates(fr, predMaps(:,:,ic-1), options, sourceType);
                end
            else
                error('Unsupported jump from type: %s', jumpFromType);
            end
            
            %ic
            predMaps(:,:,ic) = candidate2map(srcCands, [n, m], candScale);
            cands{ic} = srcCands;
        end
    end
    
    % compare
    frames = jumpFrames + after;
    indFr = find(frames <= videoLen);
    
    % save
    %save(fullfile(visRoot, sprintf('%s.mat', videos{iv})), 'frames', 'indFr', 'cands', 'predMaps');
    fprintf('%f sec\n', toc);
end