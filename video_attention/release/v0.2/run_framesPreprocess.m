% run_framesPreprocess
global dropbox;
global config;

%% settings
gbvsPath = fullfile(gdrive, 'Software', 'Saliency', 'gbvs');
addpath(gbvsPath);

uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeRawRoot = fullfile(saveRoot, 'diem');
cutsFile = fullfile(uncVideoRoot, '00_cuts.mat');
saliencyRoot = fullfile(diemDataRoot, 'frame_saliency');
gazeRoot = fullfile(diemDataRoot, 'frame_gaze');
humanRoot = fullfile(diemDataRoot, 'frame_human');
faceRoot = fullfile(diemDataRoot, 'frame_face');
motionRoot = fullfile(diemDataRoot, 'frame_motion');

calcSal = false;
calcGaze = true;
calcMotion = true;
calcFace = true;
calcPoselet = true;

%% GBVS
curDir = pwd;
cd(gbvsPath);

sc = load(cutsFile);
videos = sc.videos;

param = makeGBVSParams;
gbvs_install; % TODO stupid bug in GBVS

%% gaze
diemScreen = [1280 960];
kerRad = 52;
scaleFactor = 5;

diemScrCen = diemScreen / 2;

%% motion
alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;
ofParam = [alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations];

%% face
faceDetectorPath = fullfile(dropbox, 'Software', 'face_detector_adobe', 'win64new');
addpath(faceDetectorPath);

findfaces(fullfile(faceDetectorPath, 'Detector4.xml')); % profile and frontal

%% poselets
%%%%% configuration
poseletsRoot = fullfile(dropbox, 'Software', 'poselets');
config.CLASSES={'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'};

config.DEBUG=0; % debug and verbosity level

config.POSELET_SELECTION_SMOOTH_T = 5;

% Parameters that define the configuration distance between two patches and
% the poselet example selection algorithm
config.MIN_ROT_THRESH = pi*3/4;        % a sample that is rotated +/- more than this is discarded
config.MAX_ZOOM_THRESH = 4;            % maximum zoom for a sample to be kept (4 means each source pixel can be magnified at most x4)
config.VISUAL_DIST_WEIGHT = 0.1;       % Weight of visual distance relative to Procruster's distance
config.USE_PHOG = false;
config.USE_SEGMENT_DISTANCE = false;
config.ERR_THRESH = 2;                 % what fraction of the samples to keep per each part (based on configuration proximity)

% HOG parameters (set according to the paper N.Dalal and B.Triggs, "Histograms of Oriented Gradients
% for Human Detection" CVPR 2005)
config.HOG_CELL_DIMS = [16 16 180];
config.NUM_HOG_BINS = [2 2 9];
config.SKIN_CELL_DIMS = [2 2];
config.HOG_WTSCALE = 2;
config.HOG_NORM_EPS = 1;
config.HOG_NORM_EPS2 = 0.01;
config.HOG_NORM_MAXVAL = 0.2;
config.HOG_NO_GAUSSIAN_WEIGHT=false;
config.USE_PHOG=false;

% Scanning parameters
config.PYRAMID_SCALE_RATIO = 1.1;
config.DETECTION_IMG_MIN_NUM_PIX = 1000^2;  % if the number of pixels in a detection image is < DETECTION_IMG_SIDE^2, scales up the image to meet that threshold
config.DETECTION_IMG_MAX_NUM_PIX = 1500^2;  
config.DETECT_SVM_THRESH = 0;               % higher = more more precision, less recall
config.MAX_AGGLOMERATIVE_CLUSTER_ELEMS = 500;
config.DETECT_MAX_HITS_PER_SCALE_PER_POSELET = inf; 

% Poselet clustering parameters
config.HYP_CLUSTER_THRESH = 5; %400;    % KL-distance between poselet hits to be considered in the same cluster. Used for personalized clustering of big Qs
config.GREEDY_CLUSTER_THRESH = 5;
config.HYP_CLUSTER_MAXNUM = 100;        % Max number of clusters in an image
config.CLUSTER_HITS_CUTOFF=0.05;         % clustering threshold for bounds hypotheses
config.HYPOTHESIS_PRIOR_VAR = 1;                % value of prior on the variance of keypoint distribution
config.HYPOTHESIS_PRIOR_VARIANCE_WEIGHT = 1;    % weight of prior on the variance of keypoint distribution
config.KL_USE_WEIGHTED_DISTANCE = false;        % If using KL-divergence, do we give a separate weight for each keypoint
config.CLUSTER_BOUNDS_DIST_TYPE=0;              % type of distance metric for clustering poselets.


config.USE_MEX_HOG=true;                % disable this to use Matlab version instead of mex file for HOG
config.USE_MEX_RESIZE=false;             % disable this to use Matlab version instead of mex file for imresize

% Other parameters
config.TORSO_ASPECT_RATIO = 1.5;        % height/width of torsos
config.CROP_PREDICTED_OBJ_BOUNDS_TO_IMG=true;

for i=1:length(config.CLASSES)
    config_file = sprintf('config_%s',config.CLASSES{i});
    if exist(config_file,'file')
       config.K(i) = eval(config_file); 
    end        
end

config.DATA_DIR = fullfile(poseletsRoot, 'data');
if ~exist(config.DATA_DIR,'file')
   mkdir(config.DATA_DIR); 
end
config.COMMON_DATA_DIR = fullfile(config.DATA_DIR, 'common');
if ~exist(config.COMMON_DATA_DIR,'file')
   mkdir(config.COMMON_DATA_DIR); 
end

clear i config_file;

%%%% load the model
category = 'person';
data_root = fullfile(config.DATA_DIR, category);

faster_detection = false;  % Set this to false to run slower but higher quality

if (faster_detection)
    config.DETECTION_IMG_MIN_NUM_PIX = 500^2;  % if the number of pixels in a detection image is < DETECTION_IMG_SIDE^2, scales up the image to meet that threshold
    config.DETECTION_IMG_MAX_NUM_PIX = 750^2;  
    config.PYRAMID_SCALE_RATIO = 2;
end

% Loads the SVMs for each poselet and the Hough voting params
clear output poselet_patches fg_masks;
load(fullfile(data_root, 'model.mat')); % model
if exist('output','var')
    model=output; clear output;
end

%% run
% for iv = 1:length(videos)
% for iv = 65:65
for iv = 41:41
    fprintf('Processing %s... \n', videos{iv}); tic;
    
    vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
    meta.width = vr.Width;
    meta.height = vr.Height;
    meta.length = vr.NumberOfFrames;
    nfr = meta.length;
    
    [X, Y] = meshgrid(1:vr.Width, 1:vr.Height);
    dpts = [X(:), Y(:)]';
    frCen = [vr.Width, vr.Height] / 2;
    
    if (vr.Height < 128)
        scale = 2;
    else
        scale = 1;
    end
        
    if (calcSal)
        mkdir(saliencyRoot, videos{iv});
        salRootVideo = fullfile(saliencyRoot, videos{iv});
    end
        
    if (calcGaze) % Gaze
        mkdir(gazeRoot, videos{iv});
        gazeRootVideo = fullfile(gazeRoot, videos{iv});
        sg = load(fullfile(gazeRawRoot, sprintf('%s.mat', videos{iv})));
        [ng, ~, nfr] = size(sg.data);
        gazeData = 1/scaleFactor * (sg.data - repmat(diemScrCen, [ng, 1, nfr])) + repmat(frCen, [ng, 1, nfr]);
        clear sg;

        nfr = min(size(gazeData, 3), nfr);
    end

    if (calcPoselet)
        humans = cell(vr.NumberOfFrames, 1);
    end

    if (calcFace)
        faces = cell(vr.NumberOfFrames, 1);
    end

    if (calcMotion)
        mkdir(motionRoot, videos{iv});
        motionRootVideo = fullfile(motionRoot, videos{iv});
    end

    for ifr = nfr; % for every frame
        fr = read(vr, ifr);
        
        if (calcSal) % saliency (GBVS)
            if ((max(fr(:)) - min(fr(:))) < 25)
                saliency = zeros(vr.Height, vr.Width);
            else
                if (scale > 1)
                    fr = imresize(fr, scale);
                end
                salOut = gbvs(fr, param);
                if (scale > 1)
                    salOut.master_map_resized = imresize(salOut.master_map_resized, 1/scale);
                end
                
                saliency = salOut.master_map_resized;
            end
            
            save(fullfile(salRootVideo, sprintf('%06d.mat', ifr)), 'saliency');
        end
        
        if (calcGaze) % Gaze
            gazeLoc = gazeData(:,:,ifr);
            gazeLoc = gazeLoc(~isnan(gazeLoc(:,1)), :);
            
            if (size(gazeLoc, 1) > 0)
                % estimate KDE
                err = kerRad / scaleFactor * ones(size(gazeLoc));
                kd = kde(gazeLoc', err');
                p = evaluate(kd, dpts);
                gazeProb = reshape(p, size(X));
            else
                gazeProb = zeros(vr.Height, vr.Width);
            end
            
            save(fullfile(gazeRootVideo, sprintf('%06d.mat', ifr)), 'gazeProb');
        end
        
        if (calcPoselet) % Person detection
            [humans{ifr}, ~, ~] = detect_objects_in_image(fr, model);
        end
        
        if (calcFace) % face detector (Adobe)
            faces{ifr} = findfaces(fr);
        end
        
        if (calcMotion) % Optical flow
            if (ifr > 3)
                frp = read(vr, ifr - 2);
                [ofx, ofy] = Coarse2FineTwoFrames(fr, frp, ofParam);
                magnitude = ofx.^2 + ofy.^2;
            else
                magnitude = zeros(vr.Height, vr.Width);
                ofx = zeros(vr.Height, vr.Width);
                ofy = zeros(vr.Height, vr.Width);
            end
            
            save(fullfile(motionRootVideo, sprintf('%06d.mat', ifr)), 'magnitude', 'ofx', 'ofy');
        end
    end
    
    
    if (calcMotion) % Motion
        
    end
    
    if (calcFace)
        save(fullfile(faceRoot, sprintf('%s.mat', videos{iv})), 'meta', 'faces');
    end
    
    if (calcPoselet)
        save(fullfile(humanRoot, sprintf('%s.mat', videos{iv})), 'meta', 'humans');
    end

    fprintf('%f sec\n', toc);
end

cd(curDir);
