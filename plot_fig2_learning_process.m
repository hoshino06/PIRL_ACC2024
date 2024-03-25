% Evaluate learning process 
clear

dir_name1 = "data/RS_Naive";
dir_name2 = "data/TD20Lam0_S5000_B09";
dir_name3 = "data/TD20Lam1e-2_S5000_B09";

f = figure(1); clf(f);
f.Position(3:4) = [650 260];

hold on
plot_leaning_curves(dir_name1, [0.7, 0.7, 0.7], [0.3,0.3,0.3], 0.3, 'Original DQN')
plot_leaning_curves(dir_name2, [0.6, 0.8, 1], 'b', 0.7, 'PIRL (\lambda = 0)')
plot_leaning_curves(dir_name3, [0.7, 0.9, 0.7], [0.1, 0.5, 0.1], 0.7, 'PIRL (\lambda = 1e-2)')
hold off
uistack(findobj(gca, 'Type', 'line', 'Color', [0.3,0.3,0.3]), 'top');
uistack(findobj(gca, 'Type', 'line', 'Color', 'b'), 'top');
uistack(findobj(gca, 'Type', 'line', 'Color', [0.1, 0.5, 0.1]), 'top');
uistack(findobj(gca, 'Type', 'line', 'Color', [0.4, 0, 0.4]), 'top');

hLegend = legend('Location','northoutside','Orientation', 'horizontal');

xlabel('Episodes')
ylabel('Average Return')

xlim([0,5000])
%ylim([0,1])
fontsize(11,"points")

saveas(gcf,"data/fig2_learning_process.eps",'epsc')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plot_leaning_curves(dir_name, color_light, color_dark, FaceAlpha, disp_name)

    files = dir(fullfile(dir_name, '*.mat')); % Get a list of files
    num_file  = numel(files);

    x = cell(num_file, 1);
    y = cell(num_file, 1);
    parfor k = 1:num_file
        filename  = fullfile(dir_name, files(k).name);
        train_res = load(filename,"trainResults");
        x{k}  = train_res.trainResults.EpisodeIndex;
        y{k}  = movmean(train_res.trainResults.EpisodeReward,500);
    end
    
    % average 
    mean_data = mean(cat(2, y{:}), 2);
    plot(x{1}, mean_data, ...
         'Color', color_dark, 'LineWidth', 2, 'DisplayName', disp_name)

    % std    
    std_dev = std(cat(2, y{:}),0, 2);
    h2 = fill([x{1}', fliplr(x{1}')], [mean_data'+std_dev', fliplr(mean_data'-std_dev')], ...
              color_light, 'EdgeColor', 'none','FaceAlpha', FaceAlpha);
    hAnnotation = get(h2,'Annotation');
    hLegendEntry = get(hAnnotation,'LegendInformation');
    set(hLegendEntry,'IconDisplayStyle','off')

    % % lines
    % for i = 1:num_file
    %     h2 = plot(x{i}, y{i}, 'Color', color_light);
    %     hAnnotation = get(h2,'Annotation');
    %     hLegendEntry = get(hAnnotation,'LegendInformation');
    %     set(hLegendEntry,'IconDisplayStyle','off')
    % end    

end