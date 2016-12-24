function [posPts, posScore, negPts] = sourceCands(gazeMap, options, type)
% DEPRICATED
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

if (~exist('type', 'var'))
    type = 'random';
end

[m, n] = size(gazeMap);

if (isempty(gazeMap) || all(isnan(gazeMap(:))) || (min(gazeMap(:)) == max(gazeMap(:)))) % empty map
    posPts = [n/2, m/2];
    posScore = 1;
    negPts = [1, 1];
else
    if (strcmp(type, 'random'))
        mrg = ceil(min(m, n) / 10);
        [posPts, negPts, ~, ~] = sampleMapTopBottom(gazeMap, options.nSample, options.posPer, options.negPer, mrg);
        li = sub2ind([m, n], posPts(:,2), posPts(:,1));
        sc = gazeMap(li);
        posScore = sc(:) ./ max(sc(:));
    elseif (strcmp(type, 'rect'))
        negPts = [];
        rects = denseMap2Rects(gazeMap, options.posPer, options.rectSzTh);
        nr = size(rects, 1);
        if (nr == 0) % no rectangles
            posPts = [n/2, m/2];
            posScore = 1;
        else
            posPts = rects(:, [1, 2]) + rects(:, [3, 4]) / 2;
            posScore = zeros(nr,  1);
            
            % coverage is score
            for ir = 1:nr
                x1 = max(rects(ir, 1), 1);
                x2 = min(rects(ir, 1)+rects(ir, 3), n);
                y1 = max(rects(ir, 2), 1);
                y2 = min(rects(ir, 2)+rects(ir, 4), m);
                gzc = gazeMap(y1:y2, x1:x2);
                posScore(ir) = sum(gzc(:)) / sum(gazeMap(:));
            end
            if (max(posScore) > 0)
                posScore = posScore ./ max(posScore);
            end
        end
    else
        error('Not supported source candidate creation type: %s', type);
    end
end
