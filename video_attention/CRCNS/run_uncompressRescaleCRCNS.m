% run_uncompressRescaleCRCNS
%% settings

ffmpegRoot = 'C:\Users\dmitryr\Documents\software\ffmpeg-git-2c44aed-win64-static';
ffprobe = fullfile(ffmpegRoot, 'bin', 'ffprobe.exe');
ffmpeg = fullfile(ffmpegRoot, 'bin', 'ffmpeg.exe');
videoRoot = fullfile(crcnsRoot, 'stimuli');
outRoot = crcnsMtvRoot;
outVideoRoot = fullfile(outRoot, 'video_unc');

scales = 4; % 480 -> 120
% scales = 2:5;

%% recode videos
fid = fopen(fullfile(outVideoRoot, '00_metadata.txt'), 'w');
% format: name,originalW,originalH,scale,width,height
videos = videoListLoad(outRoot);
nv = length(videos);

for iv = 1:nv % for each video
    fprintf('Processing %s...\n', videos{iv});
    iVideo = fullfile(videoRoot, sprintf('%s.mpg', videos{iv}));
    oVideo = fullfile(outVideoRoot, sprintf('%s.avi', videos{iv}));

    % get the video size
    cmd = sprintf('%s -show_streams -i \"%s\"', ffprobe, iVideo);
    [st, res] = system(cmd);
    if (st ~= 0)
        warning('youtube:input', 'Something wrong in probing the video: %s', videos{iv});
        continue;
    end
        
    wl = regexp(res, 'width=\d*.\d*', 'match');
    hl = regexp(res, 'height=\d*.\d*', 'match');
    vw = str2double(wl{1}(7:end));
    vh = str2double(hl{1}(8:end));
    
%     % fixed scale
%     th = ceil(vh / scale);
%     tw = ceil(vw / scale);
    % adaptive scale
    hh = vh ./ scales;
    ind = find(hh > 100, 1, 'last');
    th = 4*ceil(hh(ind)/4);
    tw = 4*ceil(vw / scales(ind)/4);

    % recode video
    if (~exist(oVideo, 'file'))
%         cmd = sprintf('%s -i \"%s\" -y -vf scale=iw/%d:ih/%d -c:v rawvideo \"%s\"', ffmpeg, iVideo, scale, scale, oVideo);
        cmd = sprintf('%s -i \"%s\" -y -s %dx%d  -c:v rawvideo \"%s\"', ffmpeg, iVideo, tw, th, oVideo);
        fprintf('\tExecuting %s\n', cmd); st = 0;
        [st, ~] = system(cmd);
    end
    
    if (st ~= 0)
        warning('youtube:input', 'Something wrong in recoding of: %s', fls(iv).name);
        continue;
    end
    
    fprintf(fid, '%s,%d,%d,%d,%d,%d\n', videos{iv}, vw, vh, scales(ind), tw, th);
end

fclose(fid);
