% example_CRCNS.m
% directories
global dropbox;
dropbox = 'ROOT_OF_TOOLBOXES'; % used by configureDetectors(), assumed to have Software/poselets and Software/FaceDetect directories inside
diemDataRoot = 'YOUR_DIEM_DOWNLOAD_ROOT'; % replace
crcnsRoot = 'YOUR_CRCNS_DOWNLOAD_ROOT';  % replace
crcnsOrigRoot = fullfile(crcnsRoot, 'ORIG');
crcnsMtvRoot = fullfile(crcnsRoot, 'MTV');

% ADD ALL TOOLBOXES TO PATH HERE

% preprocess - should run once
run_convertGazeCRCNS;
precalc_GazeDataCRCNS;

% trained model is included in the distribution, but if you wish to
% re-train run the following
% precalc_GazeData;
% run_jumpTrain;

% run


