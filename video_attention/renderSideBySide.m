function outfr = renderSideBySide(infr, heatmaps, colors, cmap, sim, methodnms)
% Yonatan - Modified : 8/1/2015 added input "methodnms" to display the
% method name on the results.

[contM, contN, nmp] = size(heatmaps);

if (~exist('frameIdx', 'var'))
    frameIdx = 0;
end

if (~exist('sim', 'var'))
    sim = [];
else
    nmeas = size(sim, 2);
end

if(~exist('cmap', 'var'))
    cmap = jet(256);
end
ncol = size(cmap, 1);

if(~exist('colors', 'var'))
    colors = [1 0 0;
        0 1 0;
        0 0 1;
        1 1 0;
        1 0 1;
        0 1 1];
end

mrg = 2;
textSz = 20;

colors = colors(1:nmp, :);

switch(nmp)
    case 1
        outfr = zeros(contM + 2*mrg, contN + 2*mrg, 3);
        corI = 1;
        corJ = 1;
    case 2
        outfr = zeros(2*(contM + 2*mrg), contN + 2*mrg, 3);
        corI = [1, contM+2*mrg];
        corJ = [1, 1];
    case 3
        outfr = zeros(2*(contM + 2*mrg), 2*(contN + 2*mrg), 3);
        corI = [1, contM+2*mrg+1, 1];
        corJ = [1, 1, contN+2*mrg+1];
    case 4
        outfr = zeros(2*(contM + 2*mrg), 2*(contN + 2*mrg), 3);
        corI = [1, contM+2*mrg+1, 1, contM+2*mrg+1];
        corJ = [1, 1, contN+2*mrg+1, contN+2*mrg+1];
    case 5
        outfr = zeros(3*(contM + 2*mrg), 2*(contN + 2*mrg), 3);
        corI = [1, contM+2*mrg+1, 2*(contM+2*mrg)+1, 1, contM+2*mrg+1];
        corJ = [1, 1, 1, contN+2*mrg+1, contN+2*mrg+1];
    case 6
        outfr = zeros(3*(contM + 2*mrg), 2*(contN + 2*mrg), 3);
        corI = [1, contM+2*mrg+1, 2*(contM+2*mrg)+1, 1, contM+2*mrg+1, 2*(contM+2*mrg)+1];
        corJ = [1, 1, 1, contN+2*mrg+1, contN+2*mrg+1, contN+2*mrg+1];
    case 7
        outfr = zeros(3*(contM + 2*mrg), 3*(contN + 2*mrg), 3);
        corI = [1, contM+2*mrg+1, 2*(contM+2*mrg)+1, 1, contM+2*mrg+1, 2*(contM+2*mrg)+1, 1];
        corJ = [1, 1, 1, contN+2*mrg+1, contN+2*mrg+1, contN+2*mrg+1, 2*(contN+2*mrg)+1];
    otherwise
        error('No support for %d heat maps', nmp);
end

%TODO no support for frame yet
% render the frame
for imp = 1:nmp
    % border
    outfr(corI(imp):corI(imp)+contM+2*mrg-1, corJ(imp):corJ(imp)+contN+2*mrg-1, :) = repmat(reshape(colors(imp, :), [1 1 3]), [contM+2*mrg, contN+2*mrg, 1]);
    
    % heat map
    prob = heatmaps(:,:,imp);
    if (sum(prob(:) > 0))
        prob = prob ./ max(prob(:));
    end
    
    alphaMap = 0.7 * repmat(prob, [1 1 3]);
    rgbHM = ind2rgb(round(prob * ncol), cmap);
    gim = rgb2gray(infr);
    gim = imadjust(gim, [0; 1], [0.3 0.7]);
    gf = repmat(gim, [1 1 3]);
    [imr,imc]=size(gf);
    pos=[20,imr-20];
    gf=insertText(gf,pos,methodnms{imp},'FontSize',16);
    outfr(corI(imp)+mrg:corI(imp)+mrg+contM-1, corJ(imp)+mrg:corJ(imp)+mrg+contN-1, :) = rgbHM .* alphaMap + im2double(gf) .* (1-alphaMap);
    
    % similarity (text)
    if (~isempty(sim))
        for imeas = 1:nmeas
            if (~isnan(sim(imp, imeas)))
                bbs = [mrg, mrg+(textSz+mrg)*(imeas-1), 3*textSz, textSz + mrg, sim(imp, imeas)];
                outfr(corI(imp)+mrg:corI(imp)+mrg+contM-1, corJ(imp)+mrg:corJ(imp)+mrg+contN-1, :) = bbApply('embed', outfr(corI(imp)+mrg:corI(imp)+mrg+contM-1, corJ(imp)+mrg:corJ(imp)+mrg+contN-1, :), bbs, 'col', colors(imp, :), 'lw', 0, 'fh', textSz, 'fcol', colors(imp, :));
            end
        end
    end
end
