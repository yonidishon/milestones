% script to analyze the results in the folder
% number of candidates from each channel impact on the precision mean and
% Median hit-rate and mean distance of min distance between candidate and
% fixation point
% Date :: 19/11/2016
% Author :: Yonatan Dishon
fold = '\\cgm47\d\Dima_Analysis_Milestones\Candidates\topExp';
exps = {'My','Dima','GBVSPCAM'};

% extract datafiles for each of the exps
datafiles = cell(length(exps),1);
configs =[1:10,16];
% Full results of DIMA (CVPR2013)
Orig_results.meandist=18.295127;
Orig_results.meanHit= 81.102816;
Orig_results.medianHit=88.095238;
Orig_results.meanprec=59.199290;
% Cached GBVS+OF+CENTER cands (CVPR2013)
Orig_resultsSimple.meandist=23.762829;
Orig_resultsSimple.meanHit=77.485085;
Orig_resultsSimple.medianHit=86.111111;
Orig_resultsSimple.meanprec=69.363472;
% Recalculated results of GBVS+OF+CENTER
Orig_resultsSimpleNoC.meandist=22.321867;
Orig_resultsSimpleNoC.meanHit=79.901345;
Orig_resultsSimpleNoC.medianHit=88.372093;
Orig_resultsSimpleNoC.meanprec=69.058550;
% Cached GBVS+OF cands (CVPR2013)
Orig_resultsSimpleNoCenter.meandist=25.941708;
Orig_resultsSimpleNoCenter.meanHit=73.183575;
Orig_resultsSimpleNoCenter.medianHit=82.857143;
Orig_resultsSimpleNoCenter.meanprec=67.123563;

configs = [1:10,16,Inf];
result_all = cell(length(exps),1);
% Get all data the result_all;
for ii=1:length(exps)
    % reading all exp files and sorting them according to order
    datafiles{ii} = sort(extractfield(dir(fullfile(fold,sprintf('%s*means*.mat',exps{ii}))),'name'));
    topnum = cellfun(@(x)strsplit(x,'Top'),datafiles{ii},'UniformOutput',false);
    topnum = cellfun(@(x)strsplit(x{2},'.'),topnum,'UniformOutput',false);
    topnum = cellfun(@(x)str2num(x{1}),topnum);
    [~,k]= sort(topnum);
    exp_results = zeros(length(configs),4); % matrix of result expiriment
    for jj=1:length(configs)
        data = load(fullfile(fold,datafiles{ii}{k(jj)}));
        if isfield(data,'dstAllMean')
            exp_results(jj,:) = [data.dstAllMean,data.hitAllMean,data.prescAllMean,data.hit_AllMedian];
        elseif isfield(data,'dst_sAllMean');
           exp_results(jj,:) = [data.dst_sAllMean,data.hit_sAllMean,data.presc_sAllMean,data.hit_sAllMedian];
        else
            error('not matching fields');
        end
    end
    result_all{ii}=exp_results;
end

%% Display results
titles = {'Mean Hit rate (Solid) and Median Hit rate (Dash)','Mean Precision','Mean Distance'};

exps_str ={'PCAs+PCAm','GBVS+OF','GBVS+PCAm','CVPR2013ALL','CVPR2013GBVS+OF'};
CVPR2013 =[Orig_results.meandist,Orig_results.meanHit,Orig_results.meanprec,Orig_results.medianHit];
CVPR2013(2:end)=CVPR2013(2:end)/100;
CVPR2013GBVSOF =[Orig_resultsSimpleNoCenter.meandist,Orig_resultsSimpleNoCenter.meanHit,...
    Orig_resultsSimpleNoCenter.meanprec,Orig_resultsSimpleNoCenter.medianHit];
CVPR2013GBVSOF(2:end)=CVPR2013GBVSOF(2:end)/100;

for ii =1:length(titles) % run on each measure
    f_rate = figure('Name','Top Candidates Exp');
    hold on;
    colors = {'r','k','b'};
    for jj =1:length(exps) % plotting all exps
        if ii==1
            plot((1:length(configs))',result_all{jj}(:,2)',sprintf('%s',colors{jj}),'LineWidth',2);
            plot((1:length(configs))',result_all{jj}(:,4)',sprintf('--%s',colors{jj}),'LineWidth',2);
        elseif ii==2
            plot((1:length(configs))',result_all{jj}(:,3)',sprintf('%s',colors{jj}),'LineWidth',2);
        elseif ii==3
            plot((1:length(configs))',result_all{jj}(:,1)',sprintf('%s',colors{jj}),'LineWidth',2);
   
        end
    end
    if ii==1
        plot(5,CVPR2013(2),'og','LineWidth',2);
        plot(5,CVPR2013(4),'sg','LineWidth',2);
        plot(5,CVPR2013GBVSOF(2),'xy','LineWidth',2);
        plot(5,CVPR2013GBVSOF(4),'dy','LineWidth',2);
    elseif ii==2
        plot(5,CVPR2013(3),'og','LineWidth',2);
        plot(5,CVPR2013GBVSOF(3),'xy','LineWidth',2);
    else
        plot(5,CVPR2013(1),'og','LineWidth',2);
        plot(5,CVPR2013GBVSOF(1),'xg','LineWidth',2);
    end
    grid on;
    title(titles{ii})
    xlim([1,12]);xlabel('Number of Top Candidates from Each map');
    if ii==3
        ylabel('Pixels');
    else
        ylabel('%');
    end
    set(gca,'XTick',1:12);
    set(gca,'XTickLabel',arrayfun(@num2str,configs,'UniformOutput',false));
    if ii~=1
        legend(exps_str);
    else
        s=cell(length(exps_str)*2,1);
        s(1:2:end) = cellfun(@(x)sprintf('%s mean',x),exps_str,'UniformOutput',false);
        s(2:2:end) = cellfun(@(x)sprintf('%s median',x),exps_str,'UniformOutput',false);
        legend(s);
    end
    
    hold off;
end

