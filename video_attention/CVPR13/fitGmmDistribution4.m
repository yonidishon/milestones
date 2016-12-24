function [mu, sigma, weight, obj] = fitGmmDistribution4(prob)

obj = [];
msBw = 20;
nSmp = 10000;
regSz = 0.2;
% gfitTol = 0.01;
probTh = 0.2;

prob(prob < 0) = 0; % just in case

[m, n] = size(prob);
prob = prob ./ sum(prob(:)); % normalize
regSz = m * regSz;
mProb = max(prob(:));

[X, Y] = meshgrid(1:n, 1:m);
ismp = discretesample(prob, nSmp);

[clustCent] = MeanShiftCluster([X(ismp); Y(ismp)], msBw);
iMu = clustCent';
nG = size(iMu, 1);
mu = zeros(nG, 2);
sigma = zeros(2, 2, nG);
weight = zeros(nG, 1);
ii = 1;

% [mu,sigma,weight,iter] = fit_mix_2D_gaussian([X(ismp); Y(ismp)], nG);
% mu = mu';
% sigma = flipdim(flipdim(sigma, 2),1);

for i = 1:nG
    x0 = max(1, round(iMu(i,1) - regSz));
    y0 = max(1, round(iMu(i,2) - regSz));
    x1 = min(n, round(iMu(i,1) + regSz));
    y1 = min(m, round(iMu(i,2) + regSz));
    
    img = prob(y0:y1, x0:x1);
    [mm, nn] = size(img); 
    [X, Y] = meshgrid(1:nn, 1:mm);
    ismp = discretesample(img, round(nSmp*(regSz/m)^2));

    PeakOD = max(img(:));
    [u,covar,t,iter] = fit_mix_2D_gaussian([X(ismp); Y(ismp)], 1);
    
%     try
%         [cx,cy,sx,sy,PeakOD] = Gaussian2D(img, gfitTol);
%     catch me
%         % default gaussian
%         cx = (x1 + x0)/2;
%         cy = (y1 + y0)/2;
%         sx = (m/8)^2;
%         sy = sx;
%         PeakOD = max(img(:));
%     end
    
    if (PeakOD >= probTh * mProb)
        mu(ii, :) = [x0+u(1), y0+u(2)];
%         sigma(:,:,ii) = covar;
        sigma(:,:,ii) = rot90(covar, 2);
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

