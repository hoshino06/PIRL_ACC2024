% Evaluation of Leared Safety Probability
clear

tic
err_t20_woPDE = data_processing('data/TD20Lam0_S5000_B09');
err_t20_PDE   = data_processing('data/TD20Lam1e-2_S5000_B09');

err_t15_woPDE = data_processing('data/TD15Lam0_S5000_B09');
err_t15_PDE   = data_processing('data/TD15Lam5e-3_S5000_B09');

err_t10_woPDE = data_processing('data/TD10Lam0_S5000_B09');
err_t10_PDE   = data_processing('data/TD10Lam5e-3_S5000_B09');
toc

% % %%%% Figure 1(b) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = figure(2);  clf(f);
f.Position(3:4) = [650 250];
% 
categories = {'\tau_D = 1.0','\tau_D = 1.5','\tau_D = 2.0'};
x = categorical(categories);
x = reordercats(x,categories);
y = [err_t10_woPDE(1), err_t10_PDE(1);
     err_t15_woPDE(1), err_t15_PDE(1);
     err_t20_woPDE(1), err_t20_PDE(1);
     ];
std_dev = [
     err_t10_woPDE(2), err_t10_PDE(2);
     err_t15_woPDE(2), err_t15_PDE(2);
     err_t20_woPDE(2), err_t20_PDE(2);
    ];
% 
hold on

% bar chart
b = bar(x,y);

% error bar
[ngroups,nbars] = size(y); % Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
errorbar(x', y, std_dev, 'k', 'linestyle', 'none')
hold off

ylabel('Number of unsafe events')
ylim([1000,2000])
legend('Without PINN', 'With PINN', ...
       'Location', 'northwest')
% 
saveas(gcf,"data/fig4b_num_unsafe.eps",'epsc')
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = data_processing(dir_name)   

    files = dir(fullfile(dir_name, '*.mat')); % Get a list of files
    if isempty(files)
        error('No files found in the folder.');
    end
    
    num_unsafe_list = zeros(1, length(files));    
    parfor k = 1:length(files)
        % Construct the full file path
        filename = fullfile(dir_name, files(k).name);
        
        train_data = load(filename,"trainResults"); 
        num_unsafe = sum(train_data.trainResults.EpisodeReward == 0);
        num_unsafe_list(k) = num_unsafe;
    end
    data(1) = mean(num_unsafe_list);
    data(2) = std(num_unsafe_list);
end


function prob_list = safe_probability_MC(agent, Xini, Yini, T, n_sample)

dt = 0.1;
sigma = 0.2;

xnum  = numel(Xini);
prob_list = zeros(1, xnum);

% for initial condition
for j = 1:xnum

    % Monte Carlo simulation
    samples  = ones(1, n_sample);
    x1 = Xini(j);
    x2 = Yini(j);

    for n = 1:n_sample
    
        X = [x1; x2; T];
    
        % for each time step 
        for k = 1:T/dt
                
            % is unsafe? 
            if abs( X(2) ) > 1
                samples(n)   = 0;
                break
            end

            % calculate next state
            act_cell = agent.getAction({X});
            U = act_cell{:};
            X(1:2) = X(1:2) + dt*[( -X(1)^3 - X(2) ); ( X(1)+X(2)+ U )] ...
                    + sigma*sqrt(dt)*randn(2,1);
            X(3) = X(3) - dt;
            
        end
    end
    prob_list(j) = sum(samples)/n_sample;

end

end
