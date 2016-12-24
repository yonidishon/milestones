% cvpr13_jumpPredTest
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VERSIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
visRoot = fullfileCreate(diemDataRoot, 'vis_jump', 'jumpPred_v1');
% modelFile = fullfile(uncVideoRoot, '00_trained_model_validation_v5_3.mat'); % validation
modelFile = fullfile(uncVideoRoot, '00_jumpPred_model_cvpr13_v1.mat'); % CVPR13

jumpType = 'all'; % 'cut' or 'gaze_jump' or 'random' or 'all'
sourceType = 'cand';
% measures = {'chisq', 'auc', 'cc', 'nss'};
% methods = {'proposed', 'self', 'center', 'GBVS', 'PQFT', 'Hou'};

% cache settings
cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfile(cache.root, '00_features_jump');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = false; % use in case the feature extraction is updated
cache.renewJumps = false; % recalculate the final result

% gaze settings
gazeParam.pointSigma = 10;

% training and testing settings
testIdx = [6,8,10,11,12,14,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % used by Borji

% testSubset = 1:length(testIdx);
% testSubset = 11:length(testIdx);
testSubset = 9;
visVideo = false;

% visualization
cmap = jet(256);
colors = [1 0 0;
    0 1 0;
    0 0 1;
    1 1 0;
    1 0 1;
    0 1 1];

%% prepare
videos = videoListLoad(diemDataRoot, 'DIEM');
nv = length(videos);

% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

% load model
s = load(modelFile);
rf = s.rf;
options = s.options;
options.useLabel = false; % no need in label while testing
clear s;

nt = length(testSubset);
sim = cell(nt, 1);

vers = version('-release');
verNum = str2double(vers(1:4));

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
            dvr = VideoReader(fullfile(uncVideoRoot, sprintf('%s_depth.avi', videos{iv})));
        end
    end
    %m = vr.Height;
    %n = vr.Width;
    videoLen = vr.numberOfFrames;
    %param = struct('videoReader', vr);
    
    % load jump frames
    [jumpFrames, before, after] = jumpFramesLoad(diemDataRoot, iv, jumpType, 50, 30, videoLen - 30);
    before = -10; after = 0; % for jumps
    nc = length(jumpFrames);
    
    % load gaze data
    s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videos{iv})));
    gazeParam.gazeData = s.gaze.data;
    clear s;

    videoLen = min(videoLen, length(gazeParam.gazeData));
    
    resFile = fullfile(visRoot, sprintf('%s.mat', videos{iv}));
    if (~cache.renewJumps && exist(resFile, 'file')) % load from cache
        fprintf('(loading)... ');
        s = load(resFile);
        frames = s.frames;
        indFr = s.indFr;
        predJumps = s.predJumps;
        clear s;
        
    else % calculate
        predJumps = zeros(nc, 1);
        
        for ic = 1:nc
            if ((jumpFrames(ic) + before >= 3) && (jumpFrames(ic) + after <= videoLen))
                % preprocess frames
                srcFr = preprocessFrames(vr, dvr, jumpFrames(ic)+before, gbvsParam, ofParam, poseletModel, cache);
                dstFr = preprocessFrames(vr, dvr, jumpFrames(ic)+after, gbvsParam, ofParam, poseletModel, cache);

                % source candidates
                srcGazeMap = points2GaussMap(gazeParam.gazeData{jumpFrames(ic)+before}', ones(1, size(gazeParam.gazeData{jumpFrames(ic)+before}, 1)), 0, [n, m], gazeParam.pointSigma);
                maps = cat(3, (srcFr.ofx.^2 + srcFr.ofy.^2), srcFr.saliency);
                srcCands = jumpCandidates(srcFr.faces, srcFr.poselet_hit, maps, options);
                clear maps;
                
                % destination candidates
                dstGazeMap = points2GaussMap(gazeParam.gazeData{jumpFrames(ic)+after}', ones(1, size(gazeParam.gazeData{jumpFrames(ic)+after}, 1)), 0, [n, m], gazeParam.pointSigma);
                maps = cat(3, (dstFr.ofx.^2 + dstFr.ofy.^2), dstFr.saliency);
                dstCands = jumpCandidates(dstFr.faces, dstFr.poselet_hit, maps, options);
                
                % features
                f = jumpPredictorFeatures(srcFr, srcCands, dstFr, dstCands, options); % no cache
                clear srcFr dstFr maps;

                % predict
                if (strcmp(options.rfType, 'reg'))
                    lbl = regRF_predict(f', rf); % RF reg
                elseif (strcmp(options.rfType, 'reg-dist'))
                    lbl = regRF_predict(f', rf); % RF regression for distance
                elseif (strcmp(options.rfType, 'class'))
                    [~, lbl] = classRF_predict(f', rf); % RF classification
                    lbl = lbl(2) / sum(lbl);
                end
                
                predJumps(ic) = lbl;
            end
        end
        
        % save
        frames = jumpFrames + after;
        indFr = find(frames <= videoLen);
        save(fullfile(visRoot, sprintf('%s.mat', videos{iv})), 'frames', 'indFr', 'predJumps');
    
    end
    
%     % visualize
%     videoFile = fullfile(visRoot, sprintf('%s.avi', videos{iv}));
%     saveVideo = visVideo && (~exist(videoFile, 'file'));
%         
%     sim{i} = zeros(length(methods), length(measures), length(indFr));
%     if (saveVideo && verNum >= 2012)
%         vw = VideoWriter(videoFile, 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
%         open(vw);
%     end
%     
%     try
%         for ifr = 1:length(indFr)
%             fr = preprocessFrames(param.videoReader, frames(indFr(ifr)), gbvsParam, ofParam, poseletModel, cache);
%             [sim{i}(:,:,ifr), outMaps] = similarityFrame2(predMaps(:,:,indFr(ifr)), gazeParam.gazeData{frames(indFr(ifr))}, gazeParam.gazeData(frames(indFr([1:indFr(ifr)-1, indFr(ifr)+1:end]))), measures, ...
%                 'self', ...
%                 struct('method', 'center', 'cov', [(n/16)^2, 0; 0, (n/16)^2]), ...
%                 struct('method', 'saliency_GBVS', 'map', fr.saliency), ...
%                 struct('method', 'saliency_PQFT', 'map', fr.saliencyPqft), ...
%                 struct('method', 'saliency_Hou', 'map', fr.saliencyHou));
%             if (saveVideo && verNum >= 2012)
%                 outfr = renderSideBySide(fr.image, outMaps, colors, cmap, sim{i}(:,:,ifr));
%                 writeVideo(vw, outfr);
%             end
%         end
%     catch me
%         if (saveVideo && verNum >= 2012)
%             close(vw);
%         end
%         rethrow(me);
%     end
%     
%     if (saveVideo && verNum >= 2012)
%         close(vw);
%     end

    fprintf('%f sec\n', toc);
end

% save(fullfile(visRoot, '00_similarity.mat'), 'sim', 'measures', 'methods', 'testIdx', 'testSubset');

% %% visualize
% nmeas = length(measures);
% for im = 1:length(measures)
%     meanChiSq = nan(nt, length(methods));
%     for i = 1:nt
%         for j = 1:length(methods)
%             chiSq = sim{i}(j,im,:);
%             meanChiSq(i, j) = mean(chiSq(~isnan(chiSq)));
%         end
%     end
%     
%     ind = find(~isnan(meanChiSq(:,1)));
%     meanChiSq = meanChiSq(ind, :);
%     meanMeas = mean(meanChiSq, 1);
%     lbl = videos(testIdx(testSubset(ind)));
%     
%     % add dummy if there is only one test
%     if (size(meanChiSq, 1) == 1), meanChiSq = [meanChiSq; zeros(1, length(methods))]; end;
%     
%     figure, bar(meanChiSq);
%     imLabel(lbl, 'bottom', -90, {'FontSize',8, 'Interpreter', 'None'});
%     ylabel(measures{im});
%     title(sprintf('Mean %s', mat2str(meanMeas, 2)));
%     legend(methods);
%     
%     print('-dpng', fullfile(visRoot, sprintf('overall_%s.png', measures{im})));
% end
% 
% % histogram
% visCompareMethods(sim, methods, measures, videos, testIdx(testSubset), 'hist', visRoot);
