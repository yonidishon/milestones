% cvpr13_supVideo
%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
videoRoot = fullfile(diemDataRoot, 'video');
resRoot = fullfileCreate(diemDataRoot, 'cvpr13', 'jump_test_v6');
outRoot = fullfileCreate(diemDataRoot, 'cvpr13', 'sup_video');

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
alpha = 0.5;
candScale = 1;

% clips
clips = cell(4, 1);
clips{1}.testIdx = 9; clips{1}.frameIdx = 2085:2600; % Harry Potter
clips{2}.testIdx = 1; clips{2}.frameIdx = 190:714; % cold bloods
clips{3}.testIdx = 17; clips{3}.frameIdx = 430:943; % scramblers
clips{4}.testIdx = 19; clips{4}.frameIdx = 330:923; % univercity challenge


types = cell(5, 1);
types{1}.name = 'video'; types{1}.update = false; % just video, should be first
types{2}.name = 'ours'; types{2}.update = false; % our method
types{3}.name = 'gaze'; types{3}.update = false; % gaze
types{4}.name = 'gbvs'; types{4}.update = false; % GBVS
types{5}.name = 'pqft'; types{5}.update = false; % PQFT

%% run
[gbvsParam, ofParam, poseletModel] = configureDetectors();
f = fspecial3('average', [1 1 5]);

videos = videoListLoad(diemDataRoot);
load(fullfile(resRoot, '00_similarity.mat'), 'measures', 'methods', 'testIdx');

nv = length(clips);
nm = length(types);
ncol = size(cmap, 1);

for i = 1:nv
    iv = testSubset(clips{i}.testIdx);
    videoName = videos{testIdx(iv)};

    run = false;
    for im = 1:nm
        types{im}.file = fullfile(outRoot, sprintf('%s_%s.avi', videoName, types{im}.name));
        types{im}.save = types{im}.update || ~exist(types{im}.file, 'file');
        if (types{im}.save)
            types{im}.vw = VideoWriter(fullfile(outRoot, sprintf('%s_%s.avi', videoName, types{im}.name)), 'Motion JPEG AVI');
            open(types{im}.vw);
            run = run || types{im}.save;
        end
    end
    
    if (~run)
        fprintf('Skipping %s...\n', videoName);
        continue;
    end
    
    load(fullfile(resRoot, sprintf('%s.mat', videoName)));
    [m, n, nfr] = size(predMaps);
    
    fprintf('Processing %s... ', videoName); tic;
    
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
    predMaps = convn(predMaps, f, 'same');

    h = waitbar(0, sprintf('Video %d of %d', i, nv));

    vr = VideoReader(fullfile(videoRoot, sprintf('%s.mp4', videoName)));
    vru = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videoName)));
    
    try
        for ifr = clips{i}.frameIdx % for all requested frames
            vfr = read(vr, frames(indFr(ifr))); % read full size frame
            fr = preprocessFrames(vru, frames(indFr(ifr)), gbvsParam, ofParam, poseletModel, cache);
            gazeData.index = frames(indFr(ifr));
            [~, outMaps] = similarityFrame3(predMaps(:,:,indFr(ifr)), gazeData, measures, ...
                'self', ...
                struct('method', 'saliency_GBVS', 'map', fr.saliency), ...
                struct('method', 'saliency_PQFT', 'map', fr.saliencyPqft));
            
            % render
            mv = 0;
            for im = 1:nm
                if (types{im}.save)
                    if (strcmp(types{im}.name, 'video')) % video is special
                        mv = -1;
                        writeVideo(types{im}.vw, vfr);
                    else % render map with frame
                        prob = imresize(outMaps(:,:,im + mv), [vr.Height, vr.Width]);
                        if (sum(prob(:) > 0))
                            prob = prob ./ max(prob(:));
                        end
                        
                        alphaMap = alpha * repmat(prob, [1 1 3]);
                        rgbHM = ind2rgb(round(prob * ncol), cmap);
                        gim = rgb2gray(vfr);
                        gim = imadjust(gim, [0; 1], [0.3 0.7]);
                        gf = repmat(gim, [1 1 3]);
                        outfr = rgbHM .* alphaMap + im2double(gf) .* (1-alphaMap);
                        writeVideo(types{im}.vw, outfr);
                    end
                end
            end
            
            waitbar((ifr-clips{i}.frameIdx(1)+1) / length(clips{i}.frameIdx));
        end
    catch me
        % close videos
        for im = 1:nm
            if (types{im}.save)
                close(types{im}.vw);
            end
        end
        rethrow(me);
    end
    
    % close videos
    for im = 1:nm
        if (types{im}.save)
            close(types{im}.vw);
        end
    end
    
    close(h);
    clear vr vru;
    
    fprintf('%f sec\n', toc);
end
