function outfr = renderSideBySidePaper(infr, heatmaps, idx, colors, cmap, sim)
%

mrg = 2;
textSz = 24;
alpha = 0.5;

[contM, contN, ~] = size(heatmaps);
nmp = length(idx);

if (~exist('sim', 'var'))
    sim = [];
elseif (iscellstr(sim))
    msk = char2img(sim, textSz);
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
        0 0 1];
end

colors = colors(1:nmp, :);
outfr = zeros(3*(contM + 2*mrg), (contN + 2*mrg), 3);
corI = [1, contM+2*mrg+1, 2*(contM+2*mrg)+1];
corJ = [1, 1, 1];

% render the frame
for imp = 1:nmp
    % border
    outfr(corI(imp):corI(imp)+contM+2*mrg-1, corJ(imp):corJ(imp)+contN+2*mrg-1, :) = repmat(reshape(colors(imp, :), [1 1 3]), [contM+2*mrg, contN+2*mrg, 1]);
    
    % heat map
    prob = heatmaps(:,:,idx(imp));
    if (sum(prob(:) > 0))
        prob = prob ./ max(prob(:));
    end
    
    alphaMap = alpha * repmat(prob, [1 1 3]);
    rgbHM = ind2rgb(round(prob * ncol), cmap);
    gim = rgb2gray(infr);
    gim = imadjust(gim, [0; 1], [0.3 0.7]);
    gf = repmat(gim, [1 1 3]);
    outfr(corI(imp)+mrg:corI(imp)+mrg+contM-1, corJ(imp)+mrg:corJ(imp)+mrg+contN-1, :) = rgbHM .* alphaMap + im2double(gf) .* (1-alphaMap);
    
    % similarity (text)
    if (~isempty(sim))
        if (iscellstr(sim)) % text for every frame
            [tm, tn] = size(msk{idx(imp)});
            msk3 = repmat(msk{idx(imp)}, [1 1 3]);
            col3 = repmat(reshape(colors(imp, :), [1 1 3]), [tm tn 1]);
            outfr(corI(imp)+2*mrg:corI(imp)+2*mrg+tm-1, corJ(imp)+2*mrg:corJ(imp)+2*mrg+tn-1, :) = ...
                outfr(corI(imp)+2*mrg:corI(imp)+2*mrg+tm-1, corJ(imp)+2*mrg:corJ(imp)+2*mrg+tn-1, :) .* double(msk3) + ...
                col3 .* double(~msk3);
        else % numeric values
            for imeas = 1:nmeas
                if (~isnan(sim(imp, imeas)))
                    bbs = [mrg, mrg+(textSz+mrg)*(imeas-1), 3*textSz, textSz + mrg, sim(imp, imeas)];
                    outfr(corI(imp)+mrg:corI(imp)+mrg+contM-1, corJ(imp)+mrg:corJ(imp)+mrg+contN-1, :) = bbApply('embed', outfr(corI(imp)+mrg:corI(imp)+mrg+contM-1, corJ(imp)+mrg:corJ(imp)+mrg+contN-1, :), bbs, 'col', colors(imp, :), 'lw', 0, 'fh', textSz, 'fcol', colors(imp, :));
                end
            end
        end
    end
end
