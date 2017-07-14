function [cands, pts, score, type] = xxx_jumpCandidates3addAllPCAsPCAm(faces,humans,maps, options)
% REPLACES jumpCands
%
%   options     options for calculation, used fields:
%       useCenter       using the center candidate. 0 - do not use, 1 -
%                       always use, 2 - use only if there is no other
%       candCovScale    scaling for candidate covariance from original (1)
%       humanTh         threshold for poselet filtering
%       humanMinSzRat   detections with size smaller that maximum size 
%                       multiplied by this will be omitted
%       humanMidSz      humans below this size (relative to screen) create
%                       one candidate
%       humanTrackRat   ratio of detection to track
%       motionTh        motion threshold
%       topCandsNum     number of top candidates to use
%       minTrackSize    minimum size of the side of tracking rectangle
%
% OUTPUT
%   cands       cell array of candidate structures. Fields:
%       point       center of the candidate [x y]
%       score       score of the candidate
%       type        type of candidate: center - 1, face - 2, person - 3, 
%                   motion - 4, saliency - 5, gaze - 6 
%                   (this is according to the maps order)
%       cov         original covariance
%       rect        rectangle corresponding to the candidate [x y w h]
%       candCov     candidate covariance - used for heatmap generation
%       trackRect   tracking rectangle, [x y w h]
%   pts         all candidate points, n X 2 array. The order is the same as
%               in cands
%   score       all candidate scores, 1 X n vector
%   type        all candidate types, 1 X n vector

[h, w, nm] = size(maps);
cands = {};
candI = 1;


% faces
if (exist('faces', 'var') && ~isempty(faces))
    if faces(1)==-1
        faces={};
    else
        n = size(faces, 1);
        bbs = bbNms([faces(:,1:4), faces(:,6), ones(n, 1)], 'type', 'ms');
    
        for ic = 1:n
            xx = max(1, bbs(ic, 1));
            yy = max(1, bbs(ic, 2));
            sig = (min(bbs(ic, 3:4)) / 8)^2;
            ww = min(max(bbs(ic, 3), options.minTrackSize), w - xx);
            hh = min(max(bbs(ic, 4), options.minTrackSize), h - yy);
            %         sig = (ww / 8)^2;
            
            cands{candI}.point = [xx+ww/2, yy+hh/2];
            cands{candI}.type = 2;
            cands{candI}.score = bbs(ic, 5);
            cands{candI}.cov = [sig, 0; 0, sig];
            cands{candI}.candCov = [sig, 0; 0, sig];
            cands{candI}.trackRect = round(bbApply('resize', [xx, yy, ww, hh], options.humanTrackRat, options.humanTrackRat));
            cands{candI}.rect = round([xx, yy, ww, hh]);
            
            candI = candI + 1;
        end
    end
end

% humans
if (exist('humans', 'var') && ~isempty(humans))
    % filter detections
    i = (humans.score > options.humanTh);
    bbs = humans.bounds(:, i)';
    wts = humans.score(i);
    bbs = bbNms([bbs, wts, ones(length(wts), 1)], 'type', 'ms');

    % filter small detections
    hs = sqrt(bbs(:,3).^2 + bbs(:,4).^2);
    ihs = (hs >= options.humanMinSzRat * max(hs) & hs >= options.humanMinSz * h);
    bbs = bbs(ihs, :);
    hs = hs(ihs);
    
    n = size(bbs, 1);
    
    for ic = 1:n
        % candidates for every detection
        if (hs(ic) >= options.humanMidSz * h) % four candidates
            pts = [bbs(ic, 3)/2, bbs(ic, 4)/2; % center
                bbs(ic, 3)/4, bbs(ic, 4)/4; % left shoulder
                3*bbs(ic, 3)/4, bbs(ic, 4)/4; % right shoulder
                bbs(ic, 3)/2, bbs(ic, 3)/4]; % head
            sig = [hs(ic)/12; hs(ic)/12; hs(ic)/12; hs(ic)/6].^2;
        else % one candidate
            pts = [bbs(ic, 3)/2, bbs(ic, 4)/2];
            sig = (hs(ic)/3)^2;
        end
        
        % tracking rectangle
        xx = max(1, bbs(ic, 1));
        yy = max(1, bbs(ic, 2));
        ww = min(bbs(ic, 3), w - xx);
        hh = min(bbs(ic, 4), h - yy);
        trackRect = [xx, yy, ww, hh];
        trackRect = round(bbApply('resize', trackRect, options.humanTrackRat, options.humanTrackRat));
        
        scoreScale = [1, 0.5, 0.5, 1];
        
        for ib = 1:size(pts, 1)
            pt = [bbs(ic, 1), bbs(ic, 2)] + pts(ib, :);
            if (pt(1) >= 1 && pt(1) <= w && pt(2) >= 1 && pt(2) <= h) % inside
                cands{candI}.point = pt;
                cands{candI}.type = 3;
                cands{candI}.score = scoreScale(ib) * bbs(ic, 5);
                cands{candI}.cov = [sig(ib), 0; 0, sig(ib)];
                cands{candI}.candCov = [sig(ib), 0; 0, sig(ib)];
                cands{candI}.trackRect = trackRect;
                ax = sqrt(sig(ib));
                
                xx = max(round(pt(1)-ax), 1);
                yy = max(round(pt(2)-ax), 1);
                ww = min(round(2*ax), w - xx);
                hh = min(round(2*ax), h - yy);
                cands{candI}.rect = round([xx, yy, ww, hh]);
                
                candI = candI + 1;
            end
        end
    end
end

typeOf = 3;

% sample maps
% 1 - motion, 2 - static saliency, 3 - pcam, 4 - pcas
for i = 1:nm
    mp = maps(:,:,i);
    
% COMMENTED THIS OUT SINCE i'M USING CONTRAST NOW AND NOT MOTION MAG
%     if (i == 1) % motion
%         mp(mp < options.motionTh) = 0;
%         mp(isnan(mp)) = 0;
%     else % saliency/pca's
%         
%     end
        
    if (max(mp(:)) == min(mp(:))), continue; end;
    [mu, sigma, weight, ~] = fitGmmDistribution3(mp);
%     [mu, sigma, weight, ~] = fitGmmDistribution4(mp);
%     [mu, sigma, weight] = sampleCandidatesDistribution(mp);
    if (isempty(mu)), continue; end;
    weight = weight ./ max(weight);
    
    % filter top candidates
    if (isfield(options, 'topCandsNum') && options.topCandsNum > 0 && length(weight) > options.topCandsNum)
        [~, swi] = sort(weight, 'descend');
        ii = swi(1:options.topCandsNum);
        weight = weight(ii);
        mu = mu(ii, :);
        sigma = sigma(:,:,ii);
    end
    
    npts = size(mu, 1);
    
    for ic = 1:npts
        pt = round(mu(ic, :));
        if (pt(1) >= 1 && pt(1) <= w && pt(2) >= 1 && pt(2) <= h) % inside
            cands{candI}.point = pt;
            cands{candI}.type = typeOf+i;
            cands{candI}.cov = sigma(:, :, ic);
            cands{candI}.score = weight(ic);
            
            % candidate covariance
            cands{candI}.candCov = sigma(:, :, ic) .* options.candCovScale;
            
            [R,D,R] = svd(sigma(:, :, ic));
            normstd = sqrt( diag( D ) );
            ax = max(normstd);
            xx = max(1, round(mu(ic, 1) - ax));
            yy = max(1, round(mu(ic, 2) - ax));
            ww = min(w - xx, max(round(2*ax), options.minTrackSize));
            hh = min(h - yy, max(round(2*ax), options.minTrackSize));
            cands{candI}.trackRect = round([xx yy ww hh]);
            cands{candI}.rect = cands{candI}.trackRect;
            
            candI = candI + 1;
        end
    end
end

% center
if (options.useCenter > 0)
    ax = h/8;
    addCenter = true;
    if (options.useCenter == 2) % find close candidates
        dist = zeros(size(cands));
        cp = [w/2, h/2];
        for i = 1:length(dist)
            dist(i) = sqrt(sum((cp - cands{i}.point).^2));
        end
        
        if (min(dist) <= ax)
            addCenter = false;
        end
    end
    
    if (addCenter)
        cands{candI}.point = [w/2, h/2];
        cands{candI}.type = 1;
        cands{candI}.score = 1;
        cands{candI}.cov = [ax^2, 0; 0, ax^2];
        cands{candI}.candCov = [ax^2, 0; 0, ax^2];
        cands{candI}.trackRect = [round(w/2 - ax), round(h/2 - ax), round(2*ax), round(2*ax)];
        cands{candI}.rect = cands{candI}.trackRect;
    end
end

% assemble together (legacy)
nc = length(cands);
pts = zeros(nc, 2);
score = zeros(1, nc);
type = zeros(1, nc);

for ic = 1:nc
    pts(ic, :) = cands{ic}.point;
    score(ic) = cands{ic}.score;
    type(ic) = cands{ic}.type;
end
