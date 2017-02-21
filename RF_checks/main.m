% Script to compare different Matlab ensambles OOB performance

%fitensemble uses one of these algorithms to create an ensemble.
%For classification with two classes:
algonames={'AdaBoostM1',...
           'LogitBoost',...
           'GentleBoost',...
           'RobustBoost',...
           'LPBoost',...
           'TotalBoost',...
           'RUSBoost',...
           'Subspace',...
           'Bag'};
NumTrees=200;

% Some options for templates: 

%%%%%%% check this for more info : %%%%%%%
% https://www.mathworks.com/help/stats/templateensemble.html
% Examples:
%tTree = templateTree('MinLeafSize',20);
%templateEnsemble('AdaBoostM1',NumTrees,Tree,'LearnRate',0.1)

WeakLearner = {'Tree','Discriminant','KNN'};
ALGOStemplate=[templateEnsemble('AdaBoostM1',NumTrees,WeakLearner{1},'Type','classification','NPrint',10),...
               templateEnsemble('LogitBoost',NumTrees,WeakLearner{1},'Type','classification','NPrint',10),...
               templateEnsemble('GentleBoost',NumTrees,WeakLearner{1},'Type','classification','NPrint',10),...
               templateEnsemble('RobustBoost',NumTrees,WeakLearner{1},'Type','classification','NPrint',10),...
               templateEnsemble('LPBoost',NumTrees,WeakLearner{1},'Type','classification','NPrint',10),...
               templateEnsemble('TotalBoost',NumTrees,WeakLearner{1},'Type','classification'),...
               templateEnsemble('RUSBoost',NumTrees,WeakLearner{1},'Type','classification'),...
               templateEnsemble('Subspace',NumTrees,WeakLearner{2},'Type','classification'),...
               templateEnsemble('Bag',NumTrees,WeakLearner{1},'Type','classification')];


% Data Location:
featureRoot = '\\cgm47\D\Dima_Analysis_Milestones\ModelsFeatures\00_features_v6PCAmOFGBVS_allsrc';;
diemDataRoot = '\\cgm47\D\DIEM';

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
[Xtrain, Ytrain]=prepdata(traindata,trainIdx,featureRoot,options);
bags = cell(length(ALGOS),1);

for ii=1:length(ALGOS)
    bags(ii) = fitensemble(Xtrain,Ytrain,ALGOStemplate(ii));
end


% Plot comulative error
figure;
for ii = 1:length(ALGOS)
    plot(oobLoss(bag(ii),'mode','cumulative'));
    hold on;
end
hold off;
xlabel('Number of trees');
ylabel('Classification error');
legend(algonames,'Location','NE');