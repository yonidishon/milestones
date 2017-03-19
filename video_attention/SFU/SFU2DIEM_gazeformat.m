%SFU2DIEM gaze format

DATAROOT = '\\cgm47\d\Competition_Dataset\SFU\DATA\';
PNGROOT = 'D:\head_pose_estimation\SFUpng176x144';
DIRS = dir(DATAROOT);
ISDIRS = cell2mat(extractfield(DIRS, 'isdir'));
DIRSNM = extractfield(DIRS, 'name');
DIRSNM = DIRSNM(3:end);

for ii = 1:length(DIRSNM)
    FRMS_CNT = length(dir(fullfile(PNGROOT,DIRSNM{ii},'*.png')));
    fixmap = xxx_GetFixationsSFU(DATAROOT,DIRSNM{ii},FRMS_CNT);
    data.points = fixmap;
    save(fullfile('\\cgm47\d\Competition_Dataset\SFU','gaze',[DIRSNM{ii},'.mat']),'data');
end
