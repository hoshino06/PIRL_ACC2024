% Evaluation of Leared Safety Probability
clear

MC_calc = false;

T = 2.0; % horizon of PSC

%  mesh of initial condition
n_rez = 10;
Xlim = [-1.2, 1.2]; 
Ylim = [-0.9, 0.9]; 
xmin = Xlim(1); xmax = Xlim(2); dx = (xmax-xmin)/(n_rez-1);
ymin = Ylim(1); ymax = Ylim(2); dy = (ymax-ymin)/(n_rez-1);
[Xini,Yini] = meshgrid(xmin:dx:xmax, ymin:dy:ymax); % indeces for x and y axis

n_sample = 100;

if MC_calc == true

    tic
    err_t20_woPDE = data_processing('data/TD20Lam0_S5000_B09', Xini, Yini,n_sample);
    err_t20_PDE   = data_processing('data/TD20Lam1e-2_S5000_B09', Xini, Yini,n_sample);
    
    err_t15_woPDE = data_processing('data/TD15Lam0_S5000_B09', Xini, Yini,n_sample);
    err_t15_PDE   = data_processing('data/TD15Lam5e-3_S5000_B09', Xini, Yini,n_sample);
    
    err_t10_woPDE = data_processing('data/TD10Lam0_S5000_B09', Xini, Yini,n_sample);
    err_t10_PDE   = data_processing('data/TD10Lam3e-3_S5000_B09', Xini, Yini,n_sample);
    toc

    save data/data_fig4a_accuracy.mat err_t10_PDE err_t10_woPDE err_t15_PDE err_t15_woPDE err_t20_PDE err_t20_woPDE

else

    load data/data_fig4a_accuracy.mat err_t10_PDE err_t10_woPDE err_t15_PDE err_t15_woPDE err_t20_PDE err_t20_woPDE


end



% %%%% Figure 1(b) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = figure(1);  clf(f);
f.Position(3:4) = [650 250];
% 
x = categorical({'\tau_D = 1.0','\tau_D = 1.5','\tau_D = 2.0'});
x = reordercats(x,{'\tau_D = 1.0','\tau_D = 1.5','\tau_D = 2.0'});
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

ylabel('Mean Square Error')
%ylim([0,0.2])
legend('Without PINN', 'With PINN')
% 
saveas(gcf,"data/fig4a_unseen_regions.eps",'epsc')
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function err_data = data_processing(dir_name, Xini, Yini,n_sample)   

    files = dir(fullfile(dir_name, '*.mat')); % Get a list of files
    if isempty(files)
        error('No files found in the folder.');
    end
    
    mse_error_list = zeros(1, length(files));    
    parfor k = 1:length(files)
        % Construct the full file path
        filename = fullfile(dir_name, files(k).name);
        
        agent_data = load(filename,"agent"); 
        agent = agent_data.agent;
    
        T  = 2.0;
        % monte carlo
        prob_list_mc = safe_probability_MC(agent,Xini,Yini,T,n_sample);
    
        % critic
        prob_list_critic = zeros(1, numel(Xini));
        for i = 1:numel(Xini)    
            state = [Xini(i) Yini(i) T];
            prob_list_critic(i) = agent.getCritic.getMaxQValue({state});
        end
       
        % error
        mse_error = 2*mse(prob_list_mc,prob_list_critic); 
        mse_error_list(k) = mse_error;        
    end
    err_data(1) = mean(mse_error_list);
    err_data(2) = std(mse_error_list);

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
            if abs( X(2) ) >= 1
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
