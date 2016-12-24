function visCutJumpSideBySide(uncVideoRoot, visVideoRoot, iv, cuts, videos, varargin)

nc = length(cuts);
if (nc == 0), return; end;

cmap = jet(256);
nmp = length(varargin);
colors = [1 0 0;
        0 1 0;
        0 0 1;
        1 1 0;
        1 0 1;
        0 1 1];
colors = colors(1:nmp, :);
cutAfter = 15; % frames to sample after cut

vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
vw = VideoWriter(fullfile(visVideoRoot, sprintf('%s.avi', videos{iv})), 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
open(vw);

heatmaps = zeros(vr.Height, vr.Width, nmp);
try
    for ic = 1:nc % for every cut add frame
        if (cuts(ic) + cutAfter <= vr.NumberOfFrames)
            fr = read(vr, cuts(ic) + cutAfter);
            
            for imp = 1:nmp
                heatmaps(:,:,imp) = varargin{imp}(:,:,ic);
            end
            frOut = renderSideBySide(fr, heatmaps, colors, cmap);
            
            writeVideo(vw, frOut);
        end
    end
catch me
    close(vw);
    rethrow(me);
end

close(vw);
