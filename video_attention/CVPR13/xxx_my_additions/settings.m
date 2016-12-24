%% settings
function [] = settings()
global gdrive
global dropbox
pcaloc = '\\cgm47\D\head_pose_estimation\DIEMPCApng';
myCandssaveLoc = '\\cgm47\D\Dima_Analysis_Milestones\Candidates\MyCand_FixPCAm';

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
addpath(genpath(fullfile(gdrive, 'Software', 'Saliency', 'Hou2008'))); % Hou saliency
addpath(genpath(fullfile(gdrive,'Software','Saliency','qtfm')));%PQFT
addpath(genpath(fullfile(dropbox,'Matlab','video_attention','compare','PQFT_2'))); % pqft

addpath(genpath(fullfile(dropbox,'Software','face_detector_adobe'))); % face detector
addpath(genpath(fullfile(dropbox,'Software','poselets','code')));% poselets

addpath(genpath(fullfile(gdrive,'Software','MeanShift')));% meanShift
addpath(genpath(fullfile(gdrive,'Software','misc')));% melliecious

addpath(genpath(fullfile(gdrive,'Software','randomforest-matlab'))); %Random forest

addpath(genpath(fullfile(dropbox,'Matlab','video_attention','compare','BorjiMetrics')));
