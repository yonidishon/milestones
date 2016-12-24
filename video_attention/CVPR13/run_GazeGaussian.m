% run_GazeGaussian
%% settings
DataRoot = depthDataRoot;
gazeDataRoot = fullfile(DataRoot, 'gaze');

cache.root = fullfile(DataRoot, 'cache');
cache.frameRoot = fullfile(DataRoot, 'cache');
cache.featureRoot = fullfileCreate(cache.root, '00_features_v6');
cache.gazeRoot = fullfileCreate(cache.root, '00_gaze');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = false; % use in case the feature extraction is updated
cache.renewGaze = false; % recalculate the gaze data

nsmp = 100;
videoIdx = [1:3];
ptsSigmas = 6:1:15;

%% prepare
videos = videoListLoad(DataRoot);
nv = length(videoIdx);
nsig = length(ptsSigmas);

%% run
sim = zeros(nv, nsig);

h = waitbar(0, '');
timeSt = cputime;

for i = 1:nv
    iv = videoIdx(i);
    
    % load gaze
    s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videos{iv})));
    gazeData = s.gaze.data;
    nfr = s.gaze.length;
    m = s.gaze.height;
    n = s.gaze.width;
    clear s;

    % sample
    smp = randperm(nfr, nsmp);
    s = zeros(nsmp, nsig);
    for ip = 1:nsmp
        gazePts = gazeData{smp(ip)};
        gazePts = gazePts(~isnan(gazePts(:,1)), :);
        for isig = 1:nsig
            s(ip, isig) = gazeSelfSimilarity(gazePts, [n, m], 'chisq', ptsSigmas(isig));
        end
    end
    
    sim(i, :) = mean(s, 1);
    
    % progress
    timeC = cputime;
    timeE = (timeC - timeSt) / 60;
    timeT = timeE * nv / i;
    timeR = timeT - timeE;
    waitbar(i / nv, h, sprintf('Video %d of %d (%.2f min of %.2f min left)', i, nv, timeR, timeT));
end

close(h);

%% visualize
if (nv > 1)
    s = sim(~any(isnan(sim), 2), :);
    simm = mean(s, 1);
else
    simm = sim;
end

figure;
bar(ptsSigmas, simm);
xlabel('\sigma');
ylabel('\chi^2');
title('Gaze self-similarity');
