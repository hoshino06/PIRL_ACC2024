clear

episode_nums = [500,1000,1500];

tic
NaiveRL = get_safe_prob('data/RS_Naive', episode_nums);
PIRL    = get_safe_prob('data/TD20Lam1e-2_S5000_B09', episode_nums);
RwdShap = get_safe_prob_RS('data/RS_C005', episode_nums);
toc

% % %%%% Figure 1(b) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = figure(1);  clf(f);
f.Position(3:4) = [650 250];
% 

categories = {'500','1000','1500'};
x = categorical(categories);
x = reordercats(x,categories);
y       = [NaiveRL(1,:)' RwdShap(1,:)', PIRL(1,:)'];
std_dev = [NaiveRL(2,:)' RwdShap(2,:)', PIRL(2,:)'];

legend_names = {'Original DQN', 'Reward Shaping', 'PIRL(ours)'};

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

xlabel('Episodes')
ylabel('Safety Probability')
%ylim([0,0.2])
legend(legend_names,'Location', 'northwest')
% 
saveas(gcf,"data/fig3_reward_shaping",'epsc')
% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = get_safe_prob(dir_name, episode_nums)   

    files = dir(fullfile(dir_name, '*.mat')); % Get a list of files
    if isempty(files)
        error('No files found in the folder.');
    end

    safe_probs = zeros(length(files), length(episode_nums));    
    for k = 1:length(files)
        % Construct the full file path
        filename = fullfile(dir_name, files(k).name);
        train_data = load(filename,"trainResults"); 
        emp_probs = train_data.trainResults.AverageReward(episode_nums);
        safe_probs(k,:) = reshape(emp_probs, [1, length(emp_probs)]);
    end

    data = zeros(2, length(episode_nums));
    data(1,:) = mean(safe_probs, 1);
    data(2,:) = std(safe_probs, 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = get_safe_prob_RS(dir_name, episode_nums)   

    files = dir(fullfile(dir_name, '*.mat')); % Get a list of files
    if isempty(files)
        error('No files found in the folder.');
    end

    safe_probs = zeros(length(files), length(episode_nums));    
    for k = 1:length(files)        
        for l = 1:length(episode_nums)

            % Construct the full file path
            agent_dir = dir_name+"/agents"+num2str(k); 
            filename = fullfile(agent_dir, "Agent"+num2str(episode_nums(l))+".mat");
            agent_data = load(filename,"saved_agent"); 
            agent = agent_data.saved_agent;
            
            % Monte calro
            T = 2.0;
            n_sample = 500;
            safe_probs(k,l) = safe_probability_MC(agent, T, n_sample);

        end
    
    end

    data = zeros(2, length(episode_nums));
    data(1,:) = mean(safe_probs, 1);
    data(2,:) = std(safe_probs, 1);
end


function prob = safe_probability_MC(agent, T, n_sample)

dt = 0.1;
sigma = 0.2;

% Monte Carlo simulation
samples  = ones(1, n_sample);

for n = 1:n_sample

    s  = sign( randn([2,1]) );
    r  = rand([2, 1]);
    x1  = s(1)*( r(1) * 1.5 );
    x2  = s(2)*( r(2) * 1.0 );
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
    prob = sum(samples)/n_sample;
end

end
