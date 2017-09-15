% cvpr13_validateCands
clear all;close all;clc;
%% settings
% First guess:
global gdrive 
global dropbox
saveloc = '\\cgm47\Users\ydishon\Dropbox\school\Thesis\Video_Saliency\DocumentsByMe\ThesisDocument\AyelletLihiVer\section5requirements';
cand_test_v6 = '\\cgm47\D\DIEM\cvpr13\jump_test_v6';
%cand_test_v7 = 'D:\DIEM\cvpr13\jump_test_v7'; % This on seems to be the
%GT candidates
cand_test_v6orig = '\\cgm47\D\DimaResults\jump_test_v6_orig';
cand_test_v61 = '\\cgm47\Users\ydishon\Documents\Video_Saliency\DimaResults\jump_test_v6';

%dimaCandFold = {cand_test_v6, cand_test_v6orig,cand_test_v61};
dimaCandFold = {cand_test_v6orig};

diemDataRoot = '\\cgm47\D\DIEM';
addpath(fullfile(pwd,'xxx_my_additions'));
gdrive = '\\cgm47\D\Dimarudoy_saliency\GDrive';
dropbox = '\\cgm47\D\Dimarudoy_saliency\Dropbox';
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
%visRoot = fullfileCreate(diemDataRoot, 'cvpr13', dimaCandFold{kk});
visRoot = dimaCandFold{1};
nv = length(videoIdx);
videos = videoListLoad(diemDataRoot);
load(fullfile(visRoot, '00_similarity.mat'));

[gbvsParam, ofParam, poseletModel] = configureDetectors();

%% run
dst = cell(nv, 1);
hit = cell(nv, 1);
dst_s = cell(nv, 1);
hit_s = cell(nv, 1);
presc = cell(nv, 1);
presc_s = cell(nv, 1);
cand_hits = [];
cand_hitss = [];
dimanewcands_loc = '\\cgm47\D\Dima_Analysis_Milestones\Candidates\MyCand'; % only GBVS and OF for Dima and PCAs for me;
base_fold = '\\cgm47\D\Dima_Analysis_Milestones\ModelsFeatures\PredictionsNO_SEM';
mycands_loc = {fullfile(base_fold,'PCAmPCAsCands_10'),...
                fullfile(base_fold,'PCAmPCAsCands_16')}; 
topnums =[10,16];   
for topnum = 1:length(topnums)
fprintf('################ Top Num Exp:: %d ####################\n',topnums(topnum));
for i = 1:nv
    iv = videoIdx(i);
    videoName = videos{iv};

    fprintf('Processing %s... ', videoName); tic;

    % load candidates
%     s = load(fullfile(visRoot, sprintf('%s.mat', videoName)));
%     [m, n, nfr] = size(s.predMaps);
%     cands = s.cands;
    s = load(fullfile(dimanewcands_loc,sprintf('%s.mat', videoName)),'dimacands');
    maps = load(fullfile(dimanewcands_loc,sprintf('%s.mat', videoName)),'dimamaps');
    nfr = length(maps.dimamaps);
    [m, n,~] = size(maps.dimamaps{1});
    % Dima cands

    % my cands
    s_my = load(strjoin(fullfile(mycands_loc(topnum),sprintf('%s.mat', videoName))),'cands');
    filtcands = xxx_filtcands(s_my.cands,[4,5]);
%     
%     cands = xxx_filtcands(s_my.cands,[5]);
    % load gaze data
    ss = load(fullfile(cache.gazeRoot, sprintf('%s.mat', videoName)));
    gazeData = ss.data;
    clear ss;

%     dst{i} = zeros(nfr, 1);
%     hit{i} = zeros(nfr, 1);
    dst_s{i} = zeros(nfr, 1);
    hit_s{i} = zeros(nfr, 1);
%     presc{i} =  zeros(nfr, 1);
    presc_s{i} =  zeros(nfr, 1);
    for ifr = 1:nfr
%         [dst{i}(ifr), h ,cand_h, cand_hit_typ] = xxx_candDistGaze(gazeData.points{ifr}, cands{ifr});
        [dst_s{i}(ifr), h_s ,cand_hs,cand_hit_typ_s] = xxx_candDistGaze(gazeData.points{ifr}, filtcands{ifr});
       
%         hit{i}(ifr) = mean(h);
%         presc{i}(ifr) = cand_h;
%         cand_hits=[cand_hits,cand_hit_typ];
%         
        hit_s{i}(ifr) = mean(h_s);
        presc_s{i}(ifr) = cand_hs;
        cand_hitss=[cand_hitss,cand_hit_typ_s];

    end
    
    clear cands filtcands;
    
    fprintf('%f sec\n', toc);
end


%% results
% dstMean = zeros(nv ,1);
% hitMean = zeros(nv ,1);
dstMean_s = zeros(nv ,1);
hitMean_s = zeros(nv ,1);
% precsMean = zeros(nv,1);
precs_sMean = zeros(nv,1);

fid = fopen(fullfile(saveloc,sprintf('TopCombinedPCAsPCAm%d_data.txt',topnums(topnum))),'w');
for i = 1:nv
%     dstMean(i) = mean(dst{i}(~isnan(dst{i})));
%     hitMean(i) = mean(hit{i}(~isnan(hit{i})));
%     precsMean(i) = mean(presc{i}(~isnan(presc{i})));
%     fprintf(fid,'%s\t%fpx\t%f\t%f\n', videos{videoIdx(i)}, dstMean(i), hitMean(i),precsMean(i));
    dstMean_s(i) = mean(dst_s{i}(~isnan(dst_s{i})));
    hitMean_s(i) = mean(hit_s{i}(~isnan(hit_s{i})));
    precs_sMean(i) = mean(presc_s{i}(~isnan(presc_s{i})));
    fprintf(fid,'%s\t%fpx\t%f\t%f\n', videos{videoIdx(i)}, dstMean_s(i), hitMean_s(i),precs_sMean(i));
end

% dstAll = cell2mat(dst);
% dstAll = dstAll(~isnan(dstAll));
% dstAllMean = mean(dstAll);
% hitAll = cell2mat(hit);
% hitAll = hitAll(~isnan(hitAll));
% hitAllMean = mean(hitAll);

dst_sAll = cell2mat(dst_s);
dst_sAll = dst_sAll(~isnan(dst_sAll));
dst_sAllMean = mean(dst_sAll);
hit_sAll = cell2mat(hit_s);
hit_sAll = hit_sAll(~isnan(hit_sAll));
hit_sAllMean = mean(hit_sAll);

% prescAll = cell2mat(presc);
% prescAll = prescAll(~isnan(prescAll));
% prescAllMean = mean(prescAll);

presc_sAll = cell2mat(presc_s);
presc_sAll = presc_sAll(~isnan(presc_sAll));
presc_sAllMean = mean(presc_sAll);

% fprintf(fid,'##PCAs Candidates##\n');
% fprintf(fid,'\nMean distance: %fpx\n', dstAllMean);
% fprintf(fid,'Mean hit rate: %f%%\n', hitAllMean * 100);
% fprintf(fid,'Median hit rate: %f%%\n', median(hitAll) * 100);
% fprintf(fid,'Mean precision: %f%%\n\n', prescAllMean * 100);


fprintf(fid,'##PCAmPCAs Candidates##\n');
fprintf(fid,'\nMean distance: %fpx\n', dst_sAllMean);
fprintf(fid,'Mean hit rate: %f%%\n', hit_sAllMean * 100);
fprintf(fid,'Median hit rate: %f%%\n', median(hit_sAll) * 100);
fprintf(fid,'Mean precision: %f%%\n\n', presc_sAllMean * 100);

fs = 18;
% [h, x] = hist(dstAll, 25);
% figure, bar(x, h / sum(h));
% set(gca, 'FontSize', fs);
% xlabel('Distance, [px]');
% ylabel('Percentage');
% set(findobj(gca,'Type','text'),'FontSize',fs);
% print('-dpng', fullfile(outRoot, 'min_dist.png'));


%if kk==1 % only save once the original
% Original Cumulative histogram
% figure('Name', 'Cumu - GBVS + OF');
%     [h, x] = hist(hitAll, 25);
%     h = h / sum(h);
%     h = cumsum(h);

%     save(fullfile(saveloc,'cumhisbars_all.mat'),'x','h');
%end
% bar(x, h);
% set(gca, 'FontSize', fs);
% xlabel('Hit rate');
% ylabel('Cumulative histogram of frames per hit-rate value');
% set(findobj(gca,'Type','text'),'FontSize',fs);

%The filtered Candidates Cumulative histogram:
figure('Name', 'Cumu PCAs + PCAm');
[h, x] = hist(hit_sAll, 25);
h = h / sum(h);
h = cumsum(h);
% saving the results So I can use them later:
% save(fullfile(saveloc,sprintf('cumhisbars_%s.mat',...
%     strjoin(arrayfun(@num2str,allowedtypes, 'unif', 0),'_'))),'x','h');
bar(x, h);
set(gca, 'FontSize', fs);
xlabel('Hit rate');
ylabel('Cumulative histogram of frames per hit-rate value');
%title(sprintf('Candidates : %s', strjoin(cand_types(allowedtypes),', '))); 
set(findobj(gca,'Type','text'),'FontSize',fs);
% 
% 
% % Candidate distribution by type (1 == center , 2 == face , 3 == Poselet, 4 == OF, 5 == GBVS)
% 
% [hhh,ccc]=hist(cand_hits,unique(cand_hits));
% figure('Name','cand type in fixation hits');
% bar(1:length(ccc),hhh/sum(hhh));
% set(gca, 'XTickLabels', cand_types(ccc))
% ylabel('%')
% title('Prcentage of each Candidate type in all fixation hits');
% 
% if length(unique(cand_hitss))>1
% [hhhh,cccc]=hist(cand_hitss,unique(cand_hitss));
% figure('Name','cand (filtered) type in fixation hits');
% bar(1:length(cccc),hhhh/sum(hhhh));
% set(gca, 'XTickLabels', cand_types(cccc))
% ylabel('%')
% title({'Prcentage of each Candidate type (only low level)', 'in all fixation hits'});
% end
%print('-dpng', fullfile(outRoot, 'hit_rate.png'));
% %% visual comparison
% % videoName = 'harry_potter_6_trailer_1280x544'; frameIdx = 514;
% % videoName = 'BBC_life_in_cold_blood_1278x710'; frameIdx = 1384;
% videoName = 'DIY_SOS_1280x712'; frameIdx = 1120;
% candCol = [1 1 0];
% gazeCol = [0 1 0];
% lineWidth = 1;
% ptsSz = 3;
% outWidth = 800; 
% 
% [gbvsParam, ofParam, poseletModel] = configureDetectors();
% 
% % load model
% s = load(modelFile, 'options');
% % rf = s.rf;
% options = s.options;
% options.useLabel = false; % no need in label while testing
% clear s;
% 
% % calculate
% vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videoName)));
% fr = preprocessFrames(vr, frameIdx, gbvsParam, ofParam, poseletModel, cache);
% maps = cat(3, (fr.ofx.^2 + fr.ofy.^2), fr.saliency);
% cands = jumpCandidates3(fr.faces, fr.poselet_hit, maps, options);
% 
% % visualize
% frout = im2double(rgb2gray(fr.image));
% frout = repmat(frout, [1 1 3]);
% 
% % candidates
% frcand = repmat(reshape(candCol, [1 1 3]), [fr.height, fr.width, 1]);
% for ic = 1:length(cands)
%     msk = maskEllipse(fr.width, fr.height, cands{ic}.point, cands{ic}.cov, 1)';
%     msk = bwperim(msk, 8);
%     if (lineWidth > 1)
%         msk = imdilate(msk, strel('disk', lineWidth));
%     end
%     msk = repmat(double(msk), [1 1 3]);
%     
%     frout = (1-msk) .* frout + msk .* frcand;
% end
% 
% % gaze
% s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videoName)));
% gazePts = s.data.points{frameIdx};
% clear s;
% 
% gazePts = gazePts(~isnan(gazePts(:,1)), :);
% msk = points2binaryMap(gazePts, [fr.width, fr.height]);
% if (ptsSz > 1)
%     msk = imdilate(msk, strel('diamond', lineWidth));
% end
% msk = repmat(double(msk), [1 1 3]);
% frGaze = repmat(reshape(gazeCol, [1 1 3]), [fr.height, fr.width, 1]);
% frout = (1-msk) .* frout + msk .* frGaze;
% 
% % save
% frout = imresize(frout, [nan, outWidth]);
% figure();imshow(frout);
%imwrite(frout, fullfile(outRoot, sprintf('%s_%d_valid.png', videoName, frameIdx)), 'png');
end
restoredefaultpath;