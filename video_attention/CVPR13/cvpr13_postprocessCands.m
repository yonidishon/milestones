% cvpr13_postprocessCands
%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
visRoot = fullfileCreate(diemDataRoot, 'cvpr13', 'jump_test_v6');
modelFile = fullfile(uncVideoRoot, '00_trained_model_cvpr13_v5_3.mat'); % CVPR13

testSubset = 1:20;
% videoIdx = 9;
% videoIdx = 17;
candScale = 1;
visVideo = true;

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

%% run
videos = videoListLoad(diemDataRoot);
load(fullfile(visRoot, '00_similarity.mat'), 'sim', 'measures', 'methods', 'testIdx');

methods(strcmp(methods, 'proposed')) = {'ours'};
methods(strcmp(methods, 'self')) = {'gaze'};
    
nv = length(testSubset);
simNew = cell(nv, 1);

for i = 1:nv
    iv = testSubset(i);
    videoName = videos{testIdx(iv)};
    load(fullfile(visRoot, sprintf('%s.mat', videoName)));
    [m, n, nfr] = size(predMaps);
    
    fprintf('Postprocessing %s... ', videoName); tic;
    
    [gbvsParam, ofParam, poseletModel] = configureDetectors();
    
    % load gaze data
    s = load(fullfile(cache.gazeRoot, sprintf('%s.mat', videoName)));
    gazeData = s.data;
    clear s;
    gazeParam.gazeData = gazeData.points;
    
    % run
    for ifr = 1:nfr
        predMaps(:,:,ifr) = candidate2map(cands{ifr}, [n, m], candScale);
    end
    
    % filter
    f = fspecial3('average', [1 1 5]);
    predMaps = convn(predMaps, f, 'same');
    
    % visualize
    h = waitbar(0, sprintf('Video %d of %d', i, nv));
    
    vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videoName)));
    saveVideo = visVideo;
    
    simNew{i} = zeros(length(methods), length(measures), length(indFr));
%     roc = zeros(rocBins, length(methods), length(indFr));
    if (saveVideo)
        vw1 = VideoWriter(fullfile(visRoot, sprintf('%s_post_gbvs.avi', videoName)), 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
        vw2 = VideoWriter(fullfile(visRoot, sprintf('%s_post_pqft.avi', videoName)), 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
        open(vw1);
        open(vw2);
    end
    
    nfr = length(indFr);
    % nfr = 300;
    try
        for ifr = 1:nfr
            fr = preprocessFrames(vr, frames(indFr(ifr)), gbvsParam, ofParam, poseletModel, cache);
            gazeData.index = frames(indFr(ifr));
            [simNew{i}(:,:,ifr), outMaps] = similarityFrame3(predMaps(:,:,indFr(ifr)), gazeData, measures, ...
                'self', ...
                struct('method', 'center', 'cov', [(n/16)^2, 0; 0, (n/16)^2]), ...
                struct('method', 'saliency_GBVS', 'map', fr.saliency), ...
                struct('method', 'saliency_PQFT', 'map', fr.saliencyPqft), ...
                struct('method', 'saliency_Hou', 'map', fr.saliencyHou));
            %         [simNew(:,:,ifr), outMaps, extra] = similarityFrame2(predMaps(:,:,indFr(ifr)), gazeParam.gazeData{frames(indFr(ifr))}, gazeParam.gazeData(frames(indFr([1:indFr(ifr)-1, indFr(ifr)+1:end]))), measures, ...
            %             'self', ...
            %             struct('method', 'center', 'cov', [(n/16)^2, 0; 0, (n/16)^2]), ...
            %             struct('method', 'saliency_GBVS', 'map', fr.saliency), ...
            %             struct('method', 'saliency_PQFT', 'map', fr.saliencyPqft), ...
            %             struct('method', 'saliency_Hou', 'map', fr.saliencyHou));
            
            % store ROC
            %         for i = 1:length(extra.roc)
            %             if (~isempty(extra.roc{i}))
            %                 roc(:, i, ifr) = extra.roc{i}.rocVals(:);
            %             end
            %         end
            
            if (saveVideo)
                outfr = renderSideBySidePaper(fr.image, outMaps, [2 1 4], colors, cmap, methods);
                writeVideo(vw1, outfr);
                outfr = renderSideBySidePaper(fr.image, outMaps, [2 1 5], colors, cmap, methods);
                writeVideo(vw2, outfr);
            end
            
            waitbar(ifr / nfr);
        end
    catch me
        if (saveVideo)
            close(vw1);
            close(vw2);
            close(h);
        end
        rethrow(me);
    end
    
    if (saveVideo)
        close(vw1);
        close(vw2);
    end
    
    close(h);
    fprintf('%f sec\n', toc);
end

sim = simNew;
save(fullfile(visRoot, '00_similarity_smooth.mat'), 'sim', 'measures', 'methods', 'testIdx', 'testSubset');

%% results
visCompareMethods(sim, methods, measures, videos, testIdx(videoIdx), 'boxplot', visRoot);

% meanOld = median(sim{videoIdx}, 3);
% meanNew = median(simNew, 3);
% 
% for ims = 0:length(measures)
%     for imt = 0:length(methods)
%         if (imt == 0 && ims == 0), fprintf('\t'); continue;
%         elseif (imt == 0), fprintf('%s\t', measures{ims});
%         elseif (ims == 0), fprintf('%s\t', methods{imt});
%         else
%             fprintf('%0.2f -> %0.2f\t', meanOld(imt, ims), meanNew(imt, ims));
%         end
%     end
%     
%     fprintf('\n');
% end

% meanRoc = mean(roc, 3);
% figure;
% plot(linspace(0, 1, rocBins), meanRoc);
% legend(methods);
