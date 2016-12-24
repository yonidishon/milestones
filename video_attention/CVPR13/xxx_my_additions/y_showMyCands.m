

addpath(genpath('C:\Users\ydishon\Documents\Video_Saliency\Dimarudoy_saliency\GDrive\Software\dollar_261'));
videoIdx = [6,8,10,11,12,14,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % used by Borji;
diemDataRoot = 'D:\DIEM';

videos = videoListLoad(diemDataRoot);
nv = length(videoIdx);
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');

result_loc = 'D:\Dima_Analysis_Milestones\Candidates\MyCand_FixPCAm' ;
figure();
for ii=3:nv % run on all videos
   data = load(fullfile(result_loc,videos{videoIdx(ii)}));
   vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi',videos{videoIdx(ii)})));
   for jj = 1:length(data.frames)
       visframe = read(vr, data.frames(jj));
       types = cellfun(@(x)extractfield(x,'type'),data.mycands{jj});
       %PCAm cands
       mcands = data.mycands{jj};mcands=mcands(types==4);
       mcandvis = xxx_candsvis(mcands,visframe);
       %PCAs cands
       scands = data.mycands{jj};scands=scands(types==5);
       scandvis = xxx_candsvis(scands,visframe);
       imshow([scandvis,mcandvis]);
       drawnow;
   end
    
end