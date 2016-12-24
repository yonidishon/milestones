% This script checks the OOB error of different models learned by me or by Dimtry

% First get all the models learned that I have on the computer:
addpath(genpath('C:\Users\yoni\Documents\milestones\video_attention\yyy_toolbox'));
list_orig=subdir('\\cgm47\d\Dimtry_orig\DIEM');
list_my=subdir('\\cgm47\d\Dima_Analysis_Milestones');
list_submit=subdir('\\cgm47\D\DimaReleaseCode_CGMwebsite');
tot_list = [list_orig';list_my';list_submit'];

models =[];
for ii=1:length(tot_list)
    found = dir(sprintf('%s/00_trained_model*.mat',tot_list{ii}));
    if ~isempty(found)
        models = [models;fullfile(tot_list{ii},found.name)];
    end
end
figure('Name','OOB error rate');
    plot(model.errtr(:,1)); title('OOB error rate');  xlabel('iteration (# trees)'); ylabel('OOB error rate');