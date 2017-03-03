%clear all;close all;clc
% Script to compare different Matlab ensambles OOB performance

%fitensemble uses one of these algorithms to create an ensemble.
%For classification with two classes:
algonames={'AdaBoostM1',...
           'LogitBoost',...
           'GentleBoost',...
           'RUSBoost',... % To handle inbalance data
           'RF',...
           'RFall'};
%            'RobustBoost',...
%            'LPBoost',...
%            'TotalBoost',...
%            'RUSBoost',...
%            'Subspace',...
%            'Bag'};
NumTrees=1000;

% Some options for templates: 

%%%%%%% check this for more info : %%%%%%%
% https://www.mathworks.com/help/stats/templateensemble.html
% Examples:
%tTree = templateTree('MinLeafSize',20);
%templateEnsemble('AdaBoostM1',NumTrees,Tree,'LearnRate',0.1)

WeakLearner = {'Tree','Discriminant','KNN'};
ALGOStemplate={templateEnsemble('AdaBoostM1',NumTrees,WeakLearner{1},'Type','classification','NPrint',10),...
               templateEnsemble('LogitBoost',NumTrees,WeakLearner{1},'Type','classification','NPrint',10),...
               templateEnsemble('GentleBoost',NumTrees,WeakLearner{1},'Type','classification','NPrint',10),...
               templateEnsemble('RUSBoost',NumTrees,WeakLearner{1},'Type','classification','NPrint',10)};%,...

%                templateEnsemble('RobustBoost',NumTrees,WeakLearner{1},'Type','classification','NPrint',10),...
%                templateEnsemble('LPBoost',NumTrees,WeakLearner{1},'Type','classification','NPrint',10),...
%                templateEnsemble('TotalBoost',NumTrees,WeakLearner{1},'Type','classification'),...
%                templateEnsemble('RUSBoost',NumTrees,WeakLearner{1},'Type','classification'),...
%                templateEnsemble('Subspace',NumTrees,WeakLearner{2},'Type','classification'),...
%                templateEnsemble('Bag',NumTrees,WeakLearner{1},'Type','classification')};


% Data Location:
%featureRoot = '\\cgm47\D\Dima_Analysis_Milestones\ModelsFeatures\00_features_v6PCAmOFGBVS';
featureRoot = '\\cgm47\D\Dima_Analysis_Milestones\ModelsFeatures\00_features_v6PCAmOFGBVS_allsrc_moredata';
diemDataRoot = '\\cgm47\D\DIEM';
dstloc ='C:\Users\ydishon\Documents\milestones\RF_checks';
% Options (TODO) : 
options.featureIdx = 1:(23+9*2); % PCAmOFGBVS_allsrc - TODO - VERIFY
options.trainFeatureNum = 100000;
options.negPosRatio = 5;
options.featureLen = length(options.featureIdx);
options.videos = importdata(fullfile(diemDataRoot, 'list.txt'));
options.norm = true;

% Videos used for training
trainIdx = [1,3,5,7,9,13,17:33,35:41,43,45:47,49:52,56:58,60:69,71:73,75:82];
 
feature_file = fullfile(featureRoot,'00_total.mat');
traindata = load(feature_file);
% Training each of the trees and 

% Let us start with the same data preparation for everyone:
[X, Y]=prepdata(traindata,trainIdx,featureRoot,options);
x_red = X(:,sum(isnan(X),1)<size(X,1));
X_inpt = knnimpute(x_red,5);
%%
subsets=[0.3,0.5,1];
n=randperm(size(X_inpt,1));
for k=1:length(subsets)
    subset = n(1:int32(length(n)*subsets(k)));
    Xtemp = X_inpt(subset,:);
    Ytemp = Y(subset,:);
    fprintf('Number of samples in X = %d\n',length(subset));
    %Xtrain=Xtrain(:,[1:22,32:41]);
    cvpart = cvpartition(Ytemp,'holdout',0.2);
    Xtrain = Xtemp(training(cvpart),:);
    Ytrain = Ytemp(training(cvpart),:);
    Xtest = Xtemp(test(cvpart),:);
    Ytest = Ytemp(test(cvpart),:);
    bags = cell(5,1);%cell(length(ALGOStemplate),1);
    
    for ii=1:length(algonames)
        %     if ii==6
        %         continue;
        %         bags{ii}=[];
        %     end
        if strcmp(algonames{ii},'RF') || strcmp(algonames{ii},'RFall')
            %templS = templateTree('Surrogate','On');
            if strcmp(algonames{ii},'RF')
                 bags{ii} = TreeBagger(NumTrees,X,Y,'oobpred','on','FBoot',0.3);
            else
                bags{ii} = TreeBagger(NumTrees,X,Y,'oobpred','on');
            end
        else
            bags{ii} = fitensemble(Xtrain,Ytrain,algonames{ii},NumTrees,'Tree','Type','classification','NPrint',10,'kfold',5);
        end
    end
    save(fullfile(dstloc,sprintf('1000Inf2NaNYesfeatNormX_%d_allsrc_moredata_inp39.mat',length(subset))),'bags','Xtrain','Ytrain','Xtest','Ytest');
end
    
    
%     % Plot comulative error
%     figure;
%     for ii = 1:length(ALGOStemplate)
%         %     if ii==6
%         %         continue;
%         %     end
%         plot(kfoldLoss(bags{ii},'mode','cumulative'));
%         hold on;
%     end
%     hold off;
%     xlabel('Number of trees');
%     ylabel('Classification error');
%     legend(algonames(1:3),'Location','NE');
%     title(sprintf('Multi-source - replace Inf with NaN, Yes feature norm X length=%d',subset));
%     clear bags cvpart