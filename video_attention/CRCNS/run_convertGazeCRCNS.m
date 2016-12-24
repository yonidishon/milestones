% run_convertGazeCRCNS
%% settings
dataRoot = fullfile(crcnsRoot, 'data-mtv');
outRoot = crcnsMtvRoot;
outGazeRoot = fullfile(outRoot, 'gaze');
videoRoot = fullfile(outRoot, 'video_unc');

scale = 4; % 480 -> 120

startOffset = 271;
smpPerFrame = 8;
dataC = [639/2, 479/2]; % center of the screen in gaze tracker

% gaze -> data, height, width, length

%% prepare
s = dir(dataRoot);
ns = length(s) - 2;
subjects = cell(ns, 1);
for i = 1:ns
    subjects{i} = s(i+2).name;
end

%% convert gaze data
videos = videoListLoad(outRoot);
nv = length(videos);

for iv = 1:nv
    fprintf('Processing %s... ', videos{iv}); tic;
    
    % video data
    vr = VideoReader(fullfile(videoRoot, sprintf('%s.avi', videos{iv})));
    gaze.height = vr.Height;
    gaze.width = vr.Width;
    gaze.length = vr.NumberOfFrames;
    clear vr;
    c = [gaze.width/2, gaze.height/2];
    
    % parse gaze tracker data
    gaze.data = cell(gaze.length, 1);
    for ifr = 1:gaze.length
        gaze.data{ifr} = nan(ns, 2);
    end

    for i = 1:ns % for every subject
        gf = fullfile(dataRoot, subjects{i}, sprintf('%s.e-ceyeS', videos{iv}));
        if (exist(gf, 'file'))
            gd = importdata(gf, ' ', 3);
            
            rawData = gd.data(startOffset:end, [1,2,4]);
            nfr = min(gaze.length, floor(size(rawData, 1) / smpPerFrame));
            
            for ifr = 1:nfr
                frData = rawData((ifr-1)*smpPerFrame+1:ifr*smpPerFrame, :);
                ind = (frData(:,3) == 0 | frData(:,3) == 2); % fixations only
                if (any(ind))
                    pt = mean(frData(ind, 1:2), 1);
                    gaze.data{ifr}(i, :) = (pt - dataC) ./ scale + c;
                end
            end
        end
    end
    
    save(fullfile(outGazeRoot, sprintf('%s.mat', videos{iv})), 'gaze');
    
    fprintf('%f sec\n', toc);
end
