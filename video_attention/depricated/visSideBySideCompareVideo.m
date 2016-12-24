function [sim] = visSideBySideCompareVideo(diemDataRoot, visVideoRoot, iv, frames, predMaps, cands)

uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
rawGazeRoot = fullfile(diemDataRoot, 'gaze');

ptsSigma = 10;

[m, n, nfr] = size(predMaps);
gz = zeros(m, n, nfr);

videos = videoListLoad(diemDataRoot, 'DIEM');
s = load(fullfile(rawGazeRoot, sprintf('%s.mat', videos{iv})));
rawGaze = s.gaze;
clear s;

sim = zeros(nfr, 2);

% calculate similarity
for ifr = 1:nfr
    if (frames(ifr) <= length(rawGaze.data))
        rawGazePts = rawGaze.data{frames(ifr)};
        gz(:,:,ifr) = points2GaussMap(rawGazePts', ones(1, size(rawGazePts, 1)), 0, [n, m], ptsSigma);
    else
        rawGazePts = [];
        gz(:,:,ifr) = zeros(m ,n);
    end
    sim(ifr, 1) = probMapSimilarity(rawGazePts, predMaps(:,:,ifr), 'auc');
    sim(ifr, 2) = probMapSimilarity(gz(:,:,ifr), predMaps(:,:,ifr), 'chisq');
end

% visualize
if (~isempty(visVideoRoot))
    if (exist('cands', 'var'))
        % create candidate heatmap
        candMaps = zeros(m, n, nfr);
        fr = zeros(m, n, 3);
        for ifr = 1:nfr
            if (~isempty(cands{ifr}))
                nc = length(cands{ifr});
                rect = zeros(4, nc);
                score = zeros(1, nc);
                for ic = 1:nc
                    rect(:,ic) = cands{ifr}{ic}.trackRect;
                    score(ic) = cands{ifr}{ic}.score;
                end
                candMaps(:,:,ifr) = visRectFrame(fr, rect, score);
            end
        end
        
        visSideBySideVideo(uncVideoRoot, visVideoRoot, videos, iv, frames, predMaps, gz, candMaps);
    else
        visSideBySideVideo(uncVideoRoot, visVideoRoot, videos, iv, frames, predMaps, gz);
    end
    
    % plot
    if (~isempty(frames))
        hf = figure;
        plot(sim(:, 2));
        xlabel('Frame index'); ylabel('\chi^2 distance');
        chisq = mean(sim(~isnan(sim(:, 2)), 2));
        title(sprintf('Mean \\chi^2 distance: %f', chisq));
        fname = sprintf('%s_chisq_%d-%d.png', videos{iv}, frames(1), frames(end));
        print('-dpng', fullfile(visVideoRoot, fname));
        close(hf);
    end
end
