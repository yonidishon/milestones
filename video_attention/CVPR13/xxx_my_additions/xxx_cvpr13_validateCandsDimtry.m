% cvpr13_validateCands
clear vars;close all;clc;
%% settings
% First guess:
global gdrive 
global dropbox
%saveloc = 'D:\Dima_Analysis_Milestones\Candidates';
savloc = 'D:\Dima_Analysis_Milestones\Maps';
cand_test_v6 = 'D:\DIEM\cvpr13\jump_test_v6';
%cand_test_v7 = 'D:\DIEM\cvpr13\jump_test_v7'; % This on seems to be the
%GT candidates
cand_test_v6orig = 'C:\Users\ydishon\Documents\Video_Saliency\DimaResults\jump_test_v6_orig';
cand_test_v61 = 'C:\Users\ydishon\Documents\Video_Saliency\DimaResults\jump_test_v6';

%dimaCandFold = {cand_test_v6, cand_test_v6orig,cand_test_v61};
dimaCandFold = {cand_test_v6orig};

diemDataRoot = 'D:\DIEM';
addpath(fullfile('C:\Users\ydishon\Documents\Video_Saliency\Dimarudoy_saliency\Dropbox\Matlab\video_attention\CVPR13','xxx_my_additions'));
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

videoIdx = [6,8,10,11,12,14,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % used by Borji;
videoIdx = 14;
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

rocBins = 100;

% visualization
cmap = jet(256);
colors = [1 0 0;
    0 1 0;
    0 0 1;
    1 1 0;
    1 0 1;
    0 1 1];


%% loading
cand_types = {'Center','Faces','Poselets','OF','GBVS'};
%visRoot = fullfileCreate(diemDataRoot, 'cvpr13', dimaCandFold{kk});
visRoot = dimaCandFold{1};
nv = length(videoIdx);
videos = videoListLoad(diemDataRoot);
[gbvsParam, ofParam, poseletModel] = configureDetectors();
CHCK_CAND = 1;
VIS = 1;
textpos = [10,10];
ss = load(modelFile);
options = ss.options;
options= rmfield(options,'topCandsNum');
options= rmfield(options,'sourceCandScoreTh');
candScale = 2;
%% run
for i = 1:nv
    iv = videoIdx(i);
    videoName = videos{iv};

    fprintf('Processing %s... ', videoName); tic;

    % load candidates
    s = load(fullfile(visRoot, sprintf('%s.mat', videoName)));
    [m, n, nfr] = size(s.predMaps);
    cands = s.cands;
    
    %load gaze data
    dd = load(fullfile(gazeDataRoot, sprintf('%s.mat', videos{iv})));
    gazeParam.gazeData = dd.data.points;
    clear dd;
    
    if VIS && ~CHCK_CAND
        fig = figure('Name', 'MyFigure');
    end
    
    vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
    fig1 = figure('Name','No Cache');
    fig2 = figure('Name', 'Dimtry');
    for ifr = 1:length(s.indFr)
        % GET VISUAL FRAME
        fr = preprocessFrames(vr,  s.frames(ifr), gbvsParam, ofParam, poseletModel,cache);
        fr_nocache = preprocessFrames(vr,  s.frames(ifr), gbvsParam, ofParam, poseletModel);
        mo_mag = fr.ofx.^2+fr.ofy.^2; mo_mag = mo_mag./max(mo_mag(:));
        mo_mag_nocache = fr_nocache.ofx.^2+fr_nocache.ofy.^2; mo_mag_nocache = mo_mag_nocache./max(mo_mag_nocache(:));
        if isfield(fr,'saliencyGBVS')
            GBVS = fr.saliencyGBVS;
        else
            GBVS = fr.saliency;
        end
        diff_sal = abs(fr_nocache.saliency-GBVS);
         diff_mo = abs(mo_mag-mo_mag_nocache);
         diff_vis = abs(im2double(fr.image)-im2double(fr_nocache.image));
         fin_fr = [im2double(fr.image),repmat(GBVS,[1,1,3]),repmat(mo_mag,[1,1,3]);
                im2double(fr_nocache.image),repmat(fr_nocache.saliency,[1,1,3]),repmat(mo_mag_nocache,[1,1,3]);
                diff_vis./max(diff_vis(:)),repmat(diff_sal./max(diff_sal(:)),[1,1,3]),repmat(diff_mo./max(diff_mo(:)),[1,1,3])];
         txt = sprintf('Frame #%d, diff vis = %.2f , diff Sal =%.2f diff Motion =%.2f ', s.frames(ifr),sum(diff_vis(:)),sum(diff_sal(:)),sum(diff_mo(:)));
        if VIS && ~CHCK_CAND
            imshow(fin_fr);
            title(txt);
            pause
        elseif ~CHCK_CAND
            if ~exist(fullfile(savloc,videoName))
                mkdir(fullfile(savloc,videoName));
            end
            fin_fr = insertText(fin_fr,textpos,txt);
            imwrite(fin_fr,sprintf('%s\\%06d.png',fullfile(savloc,videoName),s.frames(ifr)));
        elseif CHCK_CAND
            %[h,w,~] = size(fr_nocache.image);
%             GazeMap = points2GaussMap(gazeParam.gazeData{s.frames(ifr)}', ones(1, size(gazeParam.gazeData{s.frames(ifr)}, 1)), 0, [n, m], gazeParam.pointSigma);
%             dummy = GazeMap;
%             dummyCand = {struct('point', [n/2, m/2], 'score', 1, 'type', 1, 'candCov', [(m/8)^2, 0; 0, (m/8)^2])}; % dummy source at center
            dummy = zeros(h,w); dummy(ceil(h/2),ceil(w/2))=1;
%             if ifr == 1
%                 dummy = candidate2map(dummyCand, [n, m], candScale);    
%             else
%                 dummy = candidate2map(cands_nocache, [n, m], candScale);   
%             end
            [cands_nocache,~,~,~] = sourceCandidates(fr_nocache,dummy,options,'cand');
            cands_cached = s.cands{ifr,1};
            check_cands = {cands_nocache;cands_cached};
            for curcands = 1:2
                types = cellfun(@(x)extractfield(x,'type'),check_cands{curcands});
                uni_types = unique(types);
                vis = cell(1,6);
                for curtype = 1:length(uni_types)
                    % FOREACH CAND TYPE GET VISUALIZATION OF all THE CANDIDATES AND THE
                    % saliency map
                    sel_ind = types == uni_types(curtype);
                    if uni_types(curtype) == 4 % motion
                        background = fr.ofx.^2+fr.ofy.^2;
                        background = background./max(background(:));
                        background = insertText(background,textpos,sprintf('Motion #:%d',sum(sel_ind)));
                    elseif uni_types(curtype) == 5 % static
                        background = GBVS;
                        background = insertText(background,textpos,sprintf('GBVS #:%d',sum(sel_ind)));
                    else % uni_types(curtype) == 2 || uni_types(curtype) == 3 %Faces and Poselets and center
                        background = fr.image;
                        background = insertText(background,textpos,sprintf('Sementic #:%d',sum(sel_ind)));
                    end
                    
                    vis{uni_types(curtype)} = xxx_candsvis(check_cands{curcands}(1,sel_ind),background);
                end
                if curcands ==1
                    figure(fig1);
                else
                    figure(fig2);
                end
                emptycells = cellfun(@isempty,vis);
                [vis{emptycells}] = deal(zeros(size(fr.image,1),size(fr.image,2),3));
                imshow([insertText(im2double(fr.image),textpos,'Input'),vis{1},vis{2};vis{3},vis{4},vis{5}]);
                if curcands == 1
                    title(sprintf('NO CACHE Frame #%d, total # of Cand:%d', s.frames(ifr),length(check_cands{curcands})));
                else
                    title(sprintf('CACHED Frame #%d, total # of Cand:%d', s.frames(ifr),length(check_cands{curcands})));
                end
            end
            %pause;
        end
        
        
    end
    
    clear cands;
    
    fprintf('%f sec\n', toc);
end


