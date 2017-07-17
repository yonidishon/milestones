% Author : Yonatan Dishon
% Date : 17/07/2017

% Script to evaluate the Ave. / Distribution of candidates number under
% different configurations
diemDataRoot = '\\cgm47\D\DIEM';
videos = videoListLoad(diemDataRoot, 'DIEM');
testIdx = [6,8,10,11,12,14,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % used by Borji
videos = videos(testIdx);
BASEFOLD='\\cgm47\D\Dima_Analysis_Milestones\ModelsFeatures\PredictionsNO_SEM';

EXPFOLDS={'PCAmGBVS',...
          'PCAmGBVS\Cands_10',...
          'PCAmGBVS\Cands_16',...
          'PCAmPCAs',...
          'PCAmPCAsCands_10',...
          'PCAmPCAsCands_16'
          };
EXPNMS={'M-PCA+GBVS (5)',...
        'M-PCA+GBVS (10)',...
        'M-PCA+GBVS (16)',...
        'M-PCA+S-PCA (5)',...
        'M-PCA+S-PCA (10)',...
        'M-PCA+S-PCA (16)'
    };      

for exps = EXPFOLDS
    fprintf('===============Exp:: %s================\n',exps{:});
    fold = fullfile(BASEFOLD,exps);
    long_vec = [];
    for vid = videos'
        data = load(fullfile(fold{:},sprintf('%s.mat',vid{:})),'cands');
        cands_num = cellfun(@(x)size(x,2),data.cands);
        fprintf('Num Frames in %s is :%d\n',vid{:} ,length(cands_num));
        long_vec = [long_vec;cands_num];
    end
    [N,edges] = histcounts(long_vec,1:max(long_vec));
    figure('Name',sprintf('%s',exps{:}));
    bar(edges(1:end-1),N);
    fprintf('[%s]\n[%s]\n',num2str(edges(1:end-1)),num2str(N))
    fprintf('===============End of Exp:: %s================\n',exps{:});
end