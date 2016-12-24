function [cands] = xxx_jumpCandidates3simpleMyCand(maps, options)
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
typeOf = 3;

% sample maps
% 1 - motion, 2 - static saliency
for i = 1:nm
    mp = maps(:,:,i);
    
    if (i == 1) % motion
        mp(isnan(mp)) = 0;
    elseif (i == 2) % saliency
        
    end
        
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
            cands{candI}.type = typeOf + i;
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

