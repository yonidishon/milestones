% set_lab_attention.m
% CGM lab @ Technion
%% directories
global dropbox;
global gdrive;
dropbox = 'C:\Users\dmitryr\Dropbox';
gdrive = 'C:\Users\dmitryr\Google Drive';
% dataRoot = 'C:\Users\dmitryr\Documents\Dima Adobe\mturk\results';
% frameRoot = 'C:\Users\dmitryr\Documents\Dima Adobe\DIEM\frames';
% saveRoot = 'C:\Users\dmitryr\Documents\Dima Adobe\mturk\save';
diemDataRoot = 'C:\Users\dmitryr\Documents\Datasets\DIEM';
crcnsRoot = 'C:\Users\dmitryr\Documents\Datasets\CRCNS-eye\CRCNS-DataShare';
% youtubeRoot = 'C:\Users\dmitryr\Documents\Dima Adobe\YouTube';
% diemRoot = 'C:\Users\dmitryr\Documents\Dima Adobe\DIEM\';

crcnsOrigRoot = fullfile(crcnsRoot, 'Dima_ORIG');
crcnsMtvRoot = fullfile(crcnsRoot, 'Dima_MTV');

%% path
% external code for Judd's method
% addpath(genpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\code_judd\'));

%% basic toolboxes
addpath(genpath(fullfile(gdrive, 'Software', 'dollar_261')));

%% toolboxes
addpath(fullfile(gdrive, 'Software', 'OpticalFlow')); % optical flow
addpath(fullfile(gdrive, 'Software', 'OpticalFlow\mex'));
addpath(fullfile(gdrive, 'Software', 'randomforest-matlab', 'RF_Class_C')); % random forests
addpath(fullfile(gdrive, 'Software', 'randomforest-matlab', 'RF_Reg_C'));

% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\kmeans');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\kstest_2s_2d');
% % addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\GMM-GMR-v2.0');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\kde2d');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\gnumex');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\kde');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\SpaceTimeSaliencyDetection');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\spectral_saliency');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\FastEMD');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\toolbox\skindetector');


% addpath(fullfile(gdrive, 'Software', 'Tracking', 'L1Track', 'L1_Tracking_v4_release')); % L1 tracking
addpath(genpath(fullfile(gdrive, 'Software', 'Tracking', 'LOT', 'LOT_Source', 'Source'))); % LOT tracking
addpath(genpath(fullfile(gdrive, 'Software', 'Saliency', 'gbvs'))); % GBVS saliency
addpath(genpath(fullfile(dropbox, 'Software', 'poselets', 'code'))); % poselets
addpath(genpath(fullfile(dropbox, 'toolbox', 'SVM-KM'))); % SVM-KM toolbox
addpath(fullfile(gdrive, 'Software', 'gmmtbx')); % GMM toolbox
addpath(fullfile(gdrive, 'Software', 'MeanShift')); % mean shift
addpath(fullfile(gdrive, 'Software', 'FitFunc')); % Gaussian fitting
addpath(fullfile(gdrive, 'Software', 'misc')); % different small codes

% Adobe code
addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\xml');
% addpath('C:\Users\dmitryr\Dropbox\Research\Dima Adobe - Code - License\MATLAB\youtube');

%% other saliency
addpath(genpath(fullfile(gdrive, 'Software', 'Saliency', 'qtfm'))); % Quaternion
addpath(fullfile(dropbox, 'Matlab', 'video_attention', 'compare', 'PQFT_2')); % PQFT
addpath(fullfile(dropbox, 'Matlab', 'video_attention', 'compare', 'BorjiMetrics')); % measures
addpath(genpath(fullfile(gdrive, 'Software', 'Saliency', 'Hou2008'))); % Hou, 2008

%% research code
addpath(fullfile(dropbox, 'Matlab'));
addpath(fullfile(dropbox, 'Matlab', 'video_attention'));
addpath(fullfile(dropbox, 'Matlab', 'video_attention', 'CRCNS'));
addpath(fullfile(dropbox, 'Matlab', 'video_attention', 'xml'));
