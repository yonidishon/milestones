function visCompareMethods(sim, methods, measures, videos, videoIdx, type, visRoot, visPaper)
%

nmeas = length(measures);
nmeth = length(methods);
nv = length(sim);

if (strcmp(type, 'hist'))
    nbins = 20;
    h = zeros(nbins, nmeth); % all the histograms

    % calculate frames
    nn = 0;
    for iv = 1:nv % for every video
        nn = nn + size(sim{iv}, 3);
    end
    
    for imeas = 1:nmeas % for each method
        % gather similarities
        simAgg = zeros(nn, nmeth);
        stid = 1;
        for iv = 1:nv % for every video
            nfr = size(sim{iv}, 3);
            simAgg(stid:stid+nfr-1, :) = squeeze(sim{iv}(:, imeas, :))';
            stid = stid + nfr;
        end
        
        % visualize
        [h, x] = hist(simAgg, nbins);
        figure;
        plot(x, h);
%         bar(x, h);
        xlabel(sprintf('Similarity (%s)', measures{imeas}));
        ylabel('Percentage');
        legend(methods);
        
        if (exist('visRoot', 'var') && ~isempty(visRoot))
            print('-dpng', fullfile(visRoot, sprintf('overall_hist_%s.png', measures{imeas})));
        end
    end
elseif (strcmp(type, 'boxplot'))
    % calculate frames
    nn = 0;
    for iv = 1:nv % for every video
        nn = nn + size(sim{iv}, 3);
    end
    
    ylbl = measures;
    ylbl(strcmp(ylbl, 'chisq')) = {'\chi^2 distance'};
    ylbl(strcmp(ylbl, 'auc')) = {'AUC similarity'};
    ylbl(strcmp(ylbl,'nss')) = {'NSS similarity'};

    for imeas = 1:nmeas % for each method
        % gather similarities
        simAgg = zeros(nn, nmeth);
        stid = 1;
        for iv = 1:nv % for every video
            nfr = size(sim{iv}, 3);
            simAgg(stid:stid+nfr-1, :) = squeeze(sim{iv}(:, imeas, :))';
            stid = stid + nfr;
        end

        % sort
        med = median(simAgg(all(~isnan(simAgg), 2), :), 1);
        if (strcmp(measures{imeas}, 'chisq'))
            [~, ord] = sort(med, 'ascend');
        else
            [~, ord] = sort(med, 'descend');
        end

        % print
        fprintf('\nMedians (%s):\n',measures{imeas});
        for io = ord
            fprintf('%s: %f\n', methods{io}, med(io));
        end
        
        % visualize
        figure;
        if (exist('visPaper', 'var') && visPaper)
            fs = 15;
            lw = 3;
            methods(strcmp(methods, 'proposed')) = {'ours'};
            methods(strcmp(methods, 'gaze')) = {'humans'};
            hl = boxplot(simAgg(:,ord), 'labels', methods(ord), 'labelorientation', 'inline', 'outliersize', 1, 'symbol', 'w');
            for ih = 1:nmeth
                set(hl(ih,:), 'LineWidth', lw);
            end
            ylabel(ylbl{imeas}, 'FontSize', fs);
            set(gca, 'FontSize', fs);
            set(findobj(gca, 'Type', 'text'), 'FontSize',fs);
        else
            boxplot(simAgg(:,ord), 'labels', methods(ord),'labelorientation','inline', 'outliersize', 1, 'symbol', 'w');
            ax = gca; ax.XTickLabelRotation = -45;
            ylabel(sprintf('Similarity (%s)', measures{imeas}));
        end
        grid on;
        if (exist('visRoot', 'var') && ~isempty(visRoot))
            print('-dpng', fullfile(visRoot, sprintf('overall_box_%s.png', measures{imeas})));
        end
    end
else
    error('Visualization type %s is not supported', type);
end
