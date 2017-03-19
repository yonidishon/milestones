%SFU png to .avi

DATAROOT = '\\cgm47\d\Competition_Dataset\SFU\DATA\';
PNGROOT = 'D:\head_pose_estimation\SFUpng176x144';
DIRS = dir(DATAROOT);
ISDIRS = cell2mat(extractfield(DIRS, 'isdir'));
DIRSNM = extractfield(DIRS, 'name');
DIRSNM = DIRSNM(3:end);
DSTFOLD = '\\cgm47\d\Competition_Dataset\SFU\avi';
for ii =1:length(DIRSNM)
    pngs = dir(fullfile(PNGROOT,DIRSNM{ii},'*.png'));
    pngsnm = extractfield(pngs,'name');
    vw = VideoWriter(fullfile(DSTFOLD,[DIRSNM{ii},'.avi']));
    open(vw);
    for jj = 1:length(pngs)
        fr = imread(fullfile(PNGROOT,DIRSNM{ii},pngsnm{jj}));
        writeVideo(vw,fr);
    end
    close(vw);
end