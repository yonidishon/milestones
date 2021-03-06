function [cands, posPts, posScore, negPts] = sourceCandidates(fr, gazeMap, options, type)
% replaces sourceCands
%
%   options     options for calculation, used fields:
%       .nSample    number of points within gaze region ('random')
%       .posPer     positive percentage
%       .negPer     negative percentage ('random')
%       .rectSzTh   rectangles below this threshold are rejected ('rect')
%   type        candidate creation type, supported
%       'random'    candidates are randomly sampled from options.posPer
%                   percentage of gaze map. options.nSample number of
%                   candidates is created
%       'rect'      candidates are created by converting the gaze map to
%                   rectangles. Points are sampled at rectangle center.
%                   This does not creates negative samples (yet)
%       'cand'      

if (~exist('type', 'var'))
    type = 'random';
end

negPts = [];
cands = {};
[m, n] = size(gazeMap);
ax = m/8;

if (isempty(gazeMap) || all(isnan(gazeMap(:))) || (min(gazeMap(:)) == max(gazeMap(:)))) % empty map
    posPts = [n/2, m/2];
    posScore = 1;
    negPts = [1, 1];
    cands{1}.point = [n/2, m/2];
    cands{1}.score = 1;
    cands{1}.type = 1;
    cands{1}.cov = [ax^2, 0; 0, ax^2];
    cands{1}.candCov = [ax^2, 0; 0, ax^2];
else
    if (strcmp(type, 'random'))
        mrg = ceil(min(m, n) / 10);
        [posPts, negPts, ~, ~] = sampleMapTopBottom(gazeMap, options.nSample, options.posPer, options.negPer, mrg);
        li = sub2ind([m, n], posPts(:,2), posPts(:,1));
        sc = gazeMap(li);
        posScore = sc(:) ./ max(sc(:));
    elseif (strcmp(type, 'rect'))
        rects = denseMap2Rects(gazeMap, options.posPer, options.rectSzTh);
        nr = size(rects, 1);
        if (nr == 0) % no rectangles
            posPts = [n/2, m/2];
            posScore = 1;
            cands{1}.point = [n/2, m/2];
            cands{1}.score = 1;
            cands{1}.type = 1;
            cands{1}.cov = [ax^2, 0; 0, ax^2];
            cands{1}.candCov = [ax^2, 0; 0, ax^2];
        else
            posPts = rects(:, [1, 2]) + rects(:, [3, 4]) / 2;
            posScore = zeros(nr,  1);
            cands = cell(nr, 1);
            
            % coverage is score
            for ir = 1:nr
                x1 = max(rects(ir, 1), 1);
                x2 = min(rects(ir, 1)+rects(ir, 3), n);
                y1 = max(rects(ir, 2), 1);
                y2 = min(rects(ir, 2)+rects(ir, 4), m);
                gzc = gazeMap(y1:y2, x1:x2);
                posScore(ir) = sum(gzc(:)) / sum(gazeMap(:));
                cands{ir}.point = posPts(ir, :);
                cands{ir}.type = 6;
                cands{ir}.cov = [(x2-x1)^2, 0; (y2-y1)^2, 0];
                cands{ir}.candCov = cands{ir}.cov;
            end
            if (max(posScore) > 0)
                posScore = posScore ./ max(posScore);
            end
            for ir = 1:nr
                cands{ir}.score = posScore(ir);
            end
        end
    elseif (strcmp(type, 'cand')) % use jump candidates
        maps = cat(3, (fr.ofx.^2 + fr.ofy.^2), fr.saliency);
        [cands, ~, ~, ~] = jumpCandidates(fr.faces, fr.poselet_hit, maps, options);
        
        % coverage is score
        nr = length(cands);
        for ir = 1:nr
            x1 = max(cands{ir}.rect(1), 1);
            x2 = min(cands{ir}.rect(1)+cands{ir}.rect(3), n);
            y1 = max(cands{ir}.rect(2), 1);
            y2 = min(cands{ir}.rect(2)+cands{ir}.rect(4), m);
            gzc = gazeMap(y1:y2, x1:x2);
            cands{ir}.candScore = cands{ir}.score;
            cands{ir}.score = sum(gzc(:)) / sum(gazeMap(:));
        end
        
        % filter low scores
        if (isfield(options, 'sourceCandScoreTh'))
            cands = filterCandidates(cands, 'scoreLow', options.sourceCandScoreTh);
        end
        
        % assemble together
        nr = length(cands);
        posPts = zeros(nr, 2);
        posScore = zeros(nr, 1);
        for ir = 1:nr
            posPts(ir, :) = cands{ir}.point;
            posScore(ir) = cands{ir}.score;
        end
    else
        error('Not supported source candidate creation type: %s', type);
    end
end
