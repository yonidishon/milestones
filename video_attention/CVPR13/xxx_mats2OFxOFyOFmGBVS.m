% Script to produce the PCAs and PCAm from DIEM movies

% .png format:
% Images are needed to be in .png format.
% file names will be in the format of #frame_PCA<s,m>.png
% each movie will have its own images in its own folder.
% folder name will be == movie name (without file extension).
addpath(fullfile('C:\Users\ydishon\Documents\milestones\video_attention\CVPR13','xxx_my_additions'));
settings();

diemDataRoot = '\\cgm47\D\DIEM';
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');

cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.renew = false;
movie_list = importdata('\\cgm47\D\DIEM\list.txt');
train_set = [7,18,20,21,22,32,39,41,46,51,56,60,65,72,73];
test_set = [6,8,10,11,12,14,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % Used by Borji on DIEM
all_set =[train_set,test_set];
dst_folder ='\\cgm47\D\head_pose_estimation\DIEMOFxOFyOFmGBVS';

[gbvsParam, ofParam, poseletModel] = configureDetectors();

for k=1:length(all_set);
    movie_name_no_ext = movie_list{all_set(k)};
    fprintf('Time is:: %s, %s Processing.... \n',datestr(datetime('now')),movie_name_no_ext);
    pca_files = dir(fullfile('\\cgm47\D\head_pose_estimation\DIEMPCApng',movie_name_no_ext,'*.png'));
    if ~exist(fullfile(dst_folder,movie_name_no_ext),'dir')
        mkdir(fullfile(dst_folder,movie_name_no_ext));
    end
    vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', movie_name_no_ext)));
    for ii=1:(length(pca_files)/2)
        fr = xxx_preprocessFramesPartial(vr, ii, gbvsParam, ofParam,cache); %TODO
        imwrite(fr.saliency,fullfile(dst_folder,movie_name_no_ext,sprintf('%06d_GBVS.png',ii)),'BitDepth',16);
        ofm=sqrt(fr.ofx.^2+fr.ofy.^2);
        if max(ofm(:))>0
            ofm=ofm./max(ofm(:));
        end
        imwrite(ofm,fullfile(dst_folder,movie_name_no_ext,sprintf('%06d_OFm.png',ii)),'BitDepth',16);
        %imwrite(fr.ofx,fullfile(dst_folder,movie_name_no_ext,sprintf('%06d_PCAm.png',ii)),'BitDepth',16);
        %imwrite(fr.ofy,fullfile(dst_folder,movie_name_no_ext,sprintf('%06d_PCAm.png',ii)),'BitDepth',16);
    end  
    fprintf('Time is:: %s,%d/%d %s Finished\n',datestr(datetime('now')),k,length(all_set),movie_name_no_ext);
end;