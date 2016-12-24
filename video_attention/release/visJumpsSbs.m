function outfr = visJumpsSbs(vr, srcFr, srcGazeMap, dstFr, dstGazeMap, jumps, options, cmap)

m = vr.Height;
n = vr.Width;
if (~exist('cmap', 'var') || isempty(cmap))
    ncol = 256;
    cmapSrc = cool(ncol);
    cmapDst = hot(ncol);
else
    ncol = size(cmap, 1);
    cmapSrc = cmap;
    cmapDst = cmap;
end

outfr = zeros(2*m, n, 3, 'uint8');
                        
% source
im = read(vr, srcFr);
gim = rgb2gray(im);
gim = im2double(imadjust(gim, [0; 1], [0.3 0.7]));
gf = repmat(gim, [1 1 3]);

% add gaze map
mg = max(max(srcGazeMap));
if (mg > 0)
    prob = srcGazeMap ./ mg;
    alphaMap = 0.7 * repmat(prob, [1 1 3]);
    rgbHM = ind2rgb(round(prob * ncol), cmapSrc);
    outfr(1:m, :, :) = im2uint8(rgbHM .* alphaMap + gf .* (1-alphaMap));
else
    outfr(1:m, :, :) = im2uint8(gf);
end

% destination
im = read(vr, dstFr);
gim = rgb2gray(im);
gim = im2double(imadjust(gim, [0; 1], [0.3 0.7]));
gf = repmat(gim, [1 1 3]);

% add gaze map
mg = max(max(dstGazeMap));
if (mg > 0)
    prob = dstGazeMap ./ mg;
    alphaMap = 0.7 * repmat(prob, [1 1 3]);
    rgbHM = ind2rgb(round(prob * ncol), cmapDst);
    outfr(m+1:2*m, :, :) = im2uint8(rgbHM .* alphaMap + gf .* (1-alphaMap));
else
    outfr(m+1:2*m, :, :) = im2uint8(gf);
end

% add circle
if (exist('options', 'var') && ~isempty(options) && isfield(options, 'gazeThreshold'))
    rects = denseMap2Rects(dstGazeMap, options.posPer, 0);
    rad = options.gazeThreshold * m;
    if (~isempty(rects))
        gtPts = rects(:, [1, 2]) + rects(:, [3, 4]) / 2;
    else
        gtPts = [];
    end

    mskCirc = false(m, n);
    for ipt = 1:size(gtPts, 1)
        msk = maskEllipse(m, n, gtPts(ipt, 2), gtPts(ipt, 1), rad, rad, 0);
        mskCirc = mskCirc + bwperim(msk, 8);
    end
    
    col = [0 1 0];
    alpha = 0.5;
    img = im2double(outfr(m+1:2*m, :, :));
    msk3 = repmat(mskCirc, [1 1 3]);
    col = repmat(reshape(col, [1 1 3]), [m, n, 1]);
    img = img .* ~msk3 + msk3 .* (img .* (1-alpha) + col .* alpha);
    outfr(m+1:2*m, :, :) = im2uint8(img);
end

% jumps
nj = size(jumps, 1);
lineMap = [255 0 0; 0 255 0];
col = zeros(nj, 1);
col(jumps(:,6) == -1) = 1;
col(jumps(:,6) == 1) = 2;
outfr = embedLine(outfr, jumps(:, [1,2]), jumps(:, [3,4]) + repmat([0, m], [nj, 1]), col, lineMap);
