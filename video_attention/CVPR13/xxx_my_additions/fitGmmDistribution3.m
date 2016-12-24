function [mu, sigma, weight, obj] = fitGmmDistribution3(prob)

% v3
obj = [];
msBw = 20;
nSmp = 10000;
regSz = 0.25;
gfitTol = 0.01;
probTh = 0.2;

prob(prob < 0) = 0; % just in case

[m, n] = size(prob);
prob = prob ./ sum(prob(:)); % normalize
regSz = m * regSz;
mProb = max(prob(:));

[X, Y] = meshgrid(1:n, 1:m);
i = discretesample(prob, nSmp);

[clustCent] = MeanShiftCluster([X(i); Y(i)], msBw);
iMu = clustCent';
nG = size(iMu, 1);
% S = struct('mu', iMu, 'Sigma', repmat(iSigma, [1 1 nG]), 'PComponents', ones(nG, 1)/nG);
% S = struct('mu', iMu, 'Sigma', repmat(iSigma, [1 2 nG]));
mu = zeros(nG, 2);
sigma = zeros(2, 2, nG);
weight = zeros(nG, 1);
ii = 1;

for i = 1:nG
    x0 = max(1, round(iMu(i,1) - regSz));
    y0 = max(1, round(iMu(i,2) - regSz));
    x1 = min(n, round(iMu(i,1) + regSz));
    y1 = min(m, round(iMu(i,2) + regSz));
    
    img = prob(y0:y1, x0:x1);
    try
        [cx,cy,sx,sy,PeakOD] = Gaussian2D(img, gfitTol);
    catch me
        % default gaussian
        cx = (x1 + x0)/2;
        cy = (y1 + y0)/2;
        sx = (m/8)^2;
        sy = sx;
        PeakOD = max(img(:));
    end
    
    if (PeakOD >= probTh * mProb)
        mu(ii, :) = [x0+cx, y0+cy];
        sigma(:,:,ii) = [sx^2, 0; 0, sy^2];
        weight(ii) = PeakOD;
        ii = ii + 1;
    end
end

if (ii == 1)
    mu = []; sigma = []; weight = [];
else
    mu = mu(1:ii-1,:);
    sigma = sigma(:,:,1:ii-1);
    weight = weight(1:ii-1);
end
