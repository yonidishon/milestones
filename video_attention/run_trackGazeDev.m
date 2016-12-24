% run_trackGazeDev
clear endFrame;

DataRoot = depthDataRoot;

%% settings
uncVideoRoot = fullfile(DataRoot, 'video_unc');
gazeDataRoot = fullfile(DataRoot, 'gaze');
visRoot = fullfile(DataRoot, 'vis_trackDev');

%% prepare
videos = videoListLoad(DataRoot, 'DIEM');

% idx = 21; startFrame = 1; endFrame = 500;
% idx = 1:84; startFrame = ones(size(idx));
idx = 1:length(videos); startFrame = ones(size(idx));
% idx = 5; startFrame = 949;

% gaze settings
gazeParam.pointSigma = 10;

gazeRectTh = 0.03; % percentage for rectangle tracking
gazeCoverTh = 0.5;

% visualization
cmap = jet(256);
ncol = size(cmap, 1);
rectCol = [1 1 1];

% cache
cache.root = fullfile(DataRoot, 'cache');
cache.trackingDeviationRoot = fullfile(cache.root, '00_tracking_deviation');


% tracking
param = loadDefaultParams;
param.inputDir = []; % to use video reader
param.debug = 0;
% param.gazeCoverTh = 0;
% param.scoreTh = 6; % normal: 2, failure: 5...10

jumpFramesAll = cell(length(idx), 1);

%% run
for i = 1:length(idx)
    iv = idx(i);
    fprintf('Working on %s...\n', videos{iv});
    
    fname = fullfile(cache.trackingDeviationRoot, sprintf('%s.mat', videos{iv}));
    if (exist(fname, 'file'))
        s = load(fname);
        jumpFramesAll{i} = s.jumpFrames;
        fprintf('\tloaded from cache\n');
        continue;
    end
%     try
        % load gaze data
        s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videos{iv})));
        gazeParam.gazeData = s.gaze.data;
        clear s;
        
        % load video
        param.videoReader = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
        if (exist('endFrame', 'var') && endFrame > 0)
            param.finFrame = min(endFrame, param.videoReader.numberOfFrames);
        else
            param.finFrame = param.videoReader.numberOfFrames;
        end
        m = param.videoReader.Height;
        n = param.videoReader.Width;
        
        fr = startFrame(i);
        % find not empty gaze
        while(isempty(gazeParam.gazeData{fr}) && fr < length(gazeParam.gazeData))
            fr = fr + 1;
        end
        
        gazeMap = zeros(m, n, param.finFrame - fr + 1);
        cands = cell(param.finFrame - fr + 1, 1);
        jumps = false(param.finFrame - fr + 1, 1);
        
        % initiate tracking
        ifr = fr;
        
        while (fr < param.finFrame && fr < length(gazeParam.gazeData))
            candI = fr - ifr + 1;
            
            jumps(candI) = true;
            
            % create rectangles from gaze map
            gazeMap(:,:,candI) = points2GaussMap(gazeParam.gazeData{fr}', ones(1, size(gazeParam.gazeData{fr}, 1)), 0, [n, m], gazeParam.pointSigma);
            rects = denseMap2Rects(gazeMap(:,:,candI), gazeRectTh);
            nr = size(rects, 1);
            fprintf('In gaze at frame %d found %d rectangles\n', fr, nr);
            
            % store candidates
            if (nr > 0) 
                cands{candI} = cell(nr, 1);
                for ir = 1:nr
                    gzc = gazeMap(rects(ir,2):rects(ir,2)+rects(ir,4), rects(ir,1):rects(ir,1)+rects(ir,3), candI);
                    
                    cands{candI}{ir}.gazeCover = sum(gzc(:)) / sum(sum(gazeMap(:,:,candI)));
                    cands{candI}{ir}.trackRect = rects(ir, :);
                    cands{candI}{ir}.score = 0;
                end
                
                % track each candidate
                param.initFrame = fr;
                ntfr = param.finFrame - fr;
                tracking = zeros(ntfr, 4, nr);
                tracksValid = false(ntfr, nr);
                trackLen = zeros(nr, 1);
                trackScore = zeros(ntfr, nr);
                gazeCover = zeros(ntfr, nr);
                
                for ir = 1:nr % for every rectangle
                    fprintf('\tTracking rectangle #%d at %s:', ir, mat2str(cands{candI}{ir}.trackRect));
                    param.target0 = cands{candI}{ir}.trackRect;
                    param.gazeCoverTh = cands{candI}{ir}.gazeCover * gazeCoverTh;
                    [target, score, gazeCoverI, gazeMapI] = LOT_trackWithGaze(param, gazeParam);
                    validIdx = 1:length(score);
                    tracking(validIdx,:,ir) = target;
                    tracksValid(validIdx,ir) = true;
                    trackLen(ir) = length(validIdx);
                    gazeCover(validIdx,ir) = gazeCoverI;
                    trackScore(validIdx,ir) = score;
                    fprintf('%d frames\n', trackLen(ir));
                end
                
                tfr = min(trackLen);
            else % no rectangles
                tfr = 1;
                cands{candI+1} = cands{candI};
                gazeMap(:,:,candI+1) = zeros(m, n);
            end
            
            % store results
            for jfr = 1:tfr
                for ir = 1:nr % for every rectangle
                    cands{candI+jfr}{ir}.trackRect = tracking(jfr, :, ir); % update rectangle
                    cands{candI+jfr}{ir}.gazeCover = gazeCover(jfr, ir); % update gaze cover
                    cands{candI+jfr}{ir}.score = trackScore(jfr, ir); % update tracking score
                end
                
                % gaze map
                gazeMap(:,:,candI+jfr) = gazeMapI(:,:,jfr);
            end
            
            fr = fr + tfr;
            
            fprintf('Tracking ended at frame %d (%d frames)\n', fr-1, tfr);
        end
        VIS = 0;
        if VIS
        % visualize
        vw = VideoWriter(fullfile(visRoot, sprintf('%s.avi', videos{iv})), 'Motion JPEG AVI');
        open(vw);
        
        try
            for fr = 1:length(cands)
                if (isempty(cands{fr})), continue; end; % skip empty
                
                im = read(param.videoReader, fr + ifr - 1);
                gim = rgb2gray(im);
                gim = im2double(imadjust(gim, [0; 1], [0.3 0.7]));
                gf = repmat(gim, [1 1 3]);
                
                % add gaze map
                mg = max(max(gazeMap(:,:,fr)));
                if (mg > 0)
                    prob = gazeMap(:,:,fr) ./ mg;
                    alphaMap = 0.7 * repmat(prob, [1 1 3]);
                    rgbHM = ind2rgb(round(prob * ncol), cmap);
                    outfr = rgbHM .* alphaMap + gf .* (1-alphaMap);
                else
                    outfr = gf;
                end
                
                outfr = im2uint8(outfr);
                
                % add candidates
                nr = length(cands{fr});
                rects = zeros(nr, 4);
                for ir = 1:nr
                    rects(ir, :) = cands{fr}{ir}.trackRect;
                end
                outfr = bbApply('embed', outfr, rects, 'col', rectCol, 'lw', 2);
                
                writeVideo(vw, outfr);
            end
        catch me
            close(vw);
            fprintf('ERROR: in writing video %s: %s\n', videos{iv}, me.message);
        end
        close(vw);
        end;
        % save jump frames
        jumpFrames = find(jumps);
        save(fname, 'jumpFrames');
        jumpFramesAll{i} = jumpFrames;
        
%     catch me
%         fprintf('ERROR at %s: %s\n', videos{iv}, me.message);
%         continue;
%     end
end

% save('jump_frames.mat', 'jumpFramesAll');
