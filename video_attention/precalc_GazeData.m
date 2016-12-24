% precalc_GazeData
%% settings
DataRoot = depthDataRoot;

gazeDataRoot = fullfile(DataRoot, 'gaze');

measures = {'chisq', 'auc'};

cache.root = fullfile(DataRoot, 'cache');
cache.frameRoot = fullfile(DataRoot, 'cache');
cache.featureRoot = fullfileCreate(cache.root, '00_features_v6');
cache.gazeRoot = fullfileCreate(cache.root, '00_gaze');
cache.renew = true; % use in case the preprocessing mechanism updated
cache.renewFeatures = true; % use in case the feature extraction is updated
cache.renewGaze = true; % recalculate the gaze data

data.pointSigma = 10;

%% prepare
videos = videoListLoad(DataRoot, 'DIEM');
nv = length(videos);

%% run
for iv = 1:nv
    fprintf('Processing %s... ', videos{iv}); tic;
    gazeFile = fullfile(cache.gazeRoot, sprintf('%s.mat', videos{iv}));
    if (~cache.renewGaze && exist(gazeFile, 'file'))
        fprintf('in cache... ');
        % check if the cache is up to date
    else
        data.meatures = measures;
        
        % load gaze data
        s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videos{iv})));
        data.points = s.gaze.data;
        data.height = s.gaze.height;
        data.width = s.gaze.width;
        clear s;
        
        nfr = length(data.points);
        data.binaryMaps = false(data.height, data.width, nfr);
        data.otherMaps = false(data.height, data.width, nfr);
        data.selfSimilarity = zeros(length(measures), nfr);
        
        wbStr = sprintf('Video %d of %d', iv, nv);
        h = waitbar(0, wbStr);
        timeSt = cputime;

        for ifr = 1:nfr % for every frame
            gazePts = data.points{ifr};
            otherGazePts = data.points([1:ifr-1, ifr+1:nfr]);
            
            % produce gaze map
            gazePts = gazePts(~isnan(gazePts(:,1)), :);
%             gazeMap = points2GaussMap(gazePts', ones(1, size(gazePts, 1)), 0, [n, m], ptsSigma);
            data.binaryMaps(:,:,ifr) = points2binaryMap(gazePts, [data.width, data.height]);
            for i = 1:length(otherGazePts)
                ogp = otherGazePts{i};
                ogp = ogp(~isnan(ogp(:,1)), :);
                data.otherMaps(:,:,ifr) = data.otherMaps(:,:,ifr) | points2binaryMap(ogp, [data.width, data.height]);
            end

            % similarity
            ss = gazeSelfSimilarity(gazePts, [data.width, data.height], measures);
            data.selfSimilarity(:, ifr) = ss(:);
            
            timeC = cputime;
            timeE = (timeC - timeSt) / 60;
            timeT = timeE * nfr / ifr;
            timeR = timeT - timeE;
            waitbar(ifr / nfr, h, sprintf('%s (%.2f min of %.2f min left)', wbStr, timeR, timeT));
        end
        
        save(gazeFile, 'data');
        close(h);
    end
    
    fprintf('%f sec\n', toc);
end
