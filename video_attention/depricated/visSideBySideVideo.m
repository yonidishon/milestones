function visSideBySideVideo(uncVideoRoot, visVideoRoot, videos, iv, frames, varargin)

nfr = length(frames);
if (nfr == 0), return; end;

cmap = jet(256);
nmp = length(varargin);
colors = [1 0 0;
        0 1 0;
        0 0 1;
        1 1 0;
        1 0 1;
        0 1 1];
colors = colors(1:nmp, :);

vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
vw = VideoWriter(fullfile(visVideoRoot, sprintf('%s.avi', videos{iv})), 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
open(vw);

heatmaps = zeros(vr.Height, vr.Width, nmp);
try
    for ifr = 1:nfr % for every frame
        fr = read(vr, frames(ifr));
        
        for imp = 1:nmp
            heatmaps(:,:,imp) = varargin{imp}(:,:,ifr);
        end
        frOut = renderSideBySide(fr, heatmaps, colors, cmap);
        
        writeVideo(vw, frOut);
    end
catch me
    close(vw);
    rethrow(me);
end

close(vw);
