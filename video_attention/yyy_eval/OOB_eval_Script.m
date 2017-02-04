% This script checks the OOB error of different models learned by me or by Dimtry

% 1. get all the models learned that I have on the computer:
addpath(genpath('C:\Users\ydishon\Documents\milestones\video_attention\yyy_toolbox'));
list_orig=subdir('\\cgm47\d\Dimtry_orig\DIEM');
list_my=subdir('\\cgm47\d\Dima_Analysis_Milestones');
%list_submit=subdir('\\cgm47\D\DimaReleaseCode_CGMwebsite');
list_submit=subdir('\\cgm47\DIEM\video_unc');
tot_list = [list_orig';list_my';list_submit'];
%tot_list = list_my';
%tot_list = [list_orig';list_my'];
%tot_list = {'\\cgm47\d\DIEM\video_unc'};

%% 2. Getting all trained models
models ={};
for ii=1:length(tot_list)
    found = dir(sprintf('%s/00_trained_model*.mat',tot_list{ii}));
    if ~isempty(found)
        for jj=1:length(found)
            models = [models;fullfile(tot_list{ii},found(jj).name)];
        end
    end
end

%% 3. Plotting the OOB error rate vs iteration;
figure('Name','OOB error rate');
%chosen_models = [models(6);models([28,31,33,34])];
%chosen_models = models([6,8:11]);
chosen_models_names = {'RO','RLL','RLLPm','PmPs-5','RLLPmPs'};
chosen_models = [models(6);models([31,33,35,36])];
flags = ones(length(chosen_models),1);
for ii=1:length(chosen_models)
    model = load(chosen_models{ii});
    if isfield(model.rf,'errtr')
        plot(model.rf.errtr(:,1));
    else
        fprintf('Model:: %s has no OOB error\n',chosen_models{ii});
        flags(ii)=0;
        continue;
    end
    hold on;
    drawnow;
    grid on;
end
hold off;
title('OOB error rate');
xlabel('iteration (# trees)');
ylabel('OOB error rate');
%legend(chosen_models(boolean(flags)),'Interpreter', 'none');
legend(chosen_models_names);
%% 4. Getting the importantcy of each feature
for ii=1:length(chosen_models)
    figure('Name',chosen_models{ii})
    model = load(chosen_models{ii});
    bar(model.rf.importance(:,1));xlabel('feature');ylabel('magnitude');
    title('Feature Improtance');   
    drawnow;
    grid on;
end

