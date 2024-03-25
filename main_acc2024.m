%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train PIRL agent for planer example
function [] = main_acc2024(AgentIdx, AgentDir)

    if (nargin < 1), clear; AgentIdx = NaN; AgentDir = NaN; end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Environment (defined at the bottom)
obsInfo = rlNumericSpec([3,1]);
actInfo = rlFiniteSetSpec(-1:0.5:1);
env = rlFunctionEnv(obsInfo,actInfo, @stepFunction, @resetFunction);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Custom DQN Agent
numNeurons = 32; % 256
layers = [
    featureInputLayer(prod(obsInfo.Dimension), Name="featureinput")
    fullyConnectedLayer(numNeurons,Name="fc1")
    tanhLayer(Name="tanh_1")
    fullyConnectedLayer(numNeurons,Name="fc2")
    tanhLayer(Name="tanh_2")
    fullyConnectedLayer(numNeurons,Name="fc3")
    tanhLayer(Name="tanh_3")
    fullyConnectedLayer(numel(actInfo.Elements), Name="output")
    ];
Critic = rlVectorQValueFunction( dlnetwork(layers), obsInfo, actInfo );

AgentOptions = rlDQNAgentOptions;
AgentOptions.DiscountFactor = 1.00; %(default:0.99)
AgentOptions.MiniBatchSize  =   64; %(default:64)
AgentOptions.EpsilonGreedyExploration.EpsilonDecay = 0.0002; %(default:0.005)
AgentOptions.EpsilonGreedyExploration.EpsilonMin = 0.01; %(default:0.01)

%agent = rlDQNAgent(obsInfo,actInfo, AgentOptions);
agent = customDQNAgent(Critic, AgentOptions);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
trainOpts = rlTrainingOptions(...
    MaxEpisodes = 5000,...
    ScoreAveragingWindowLength = 100, ...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=1.1, ...
    Plots= "none");

% Plot if single agent trained
if isnan(AgentIdx) 
    trainOpts.Plots = "training-progress"; % "none"; % 
end

% % Parallelization didn't improved training speed
% trainOpts.UseParallel = true;
% trainOpts.ParallelizationOptions.Mode = "async";

do_training = true;
if do_training == true
    if isnan(AgentIdx)
        trainResults = train(agent,env,trainOpts);
        save data_trial/data.mat agent env trainResults
    else
        disp("Training Agent # "+num2str(AgentIdx))
        agent_dir = "data/"+AgentDir;
        if ~exist(agent_dir, 'dir'), mkdir(agent_dir); end

        trainResults = train(agent,env,trainOpts);

        file_name = agent_dir+"/data"+num2str(AgentIdx)+".mat";
        save (file_name, "agent", "env", "trainResults")
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementations of environment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [InitialObservation, LoggedSignal] = resetFunction()

    s  = sign( randn([2,1]) );
    r  = rand([2, 1]);

    X1  = s(1)*( r(1) * 1.5 );
    X2  = s(2)*( r(2) * 1.0 );

    TD  = 2.0;

    LoggedSignal.State = [X1; X2; TD];
    InitialObservation = LoggedSignal.State;

end

function [NextObs,Reward,IsDone,LoggedSignals] = stepFunction(Action,LoggedSignals)

    % Parameters
    dt = 0.1;
    sigma = 0.2;

    % Update state
    X     = LoggedSignals.State(1:2);
    U     = Action;

    X_nxt = X + dt*[( -X(1)^3 - X(2) ); ( X(1)+X(2)+ U )] ...
            + sigma*sqrt(dt)*randn(2,1);


    T = LoggedSignals.State(end) - dt;
    LoggedSignals.State = [X_nxt; T];

    % Transform state to observation. 
    NextObs = LoggedSignals.State;

    % Check terminal condition.
    IsTimeOver = T <= 0;
    IsUnsafe   = abs( X(2) ) >= 1 ;
    IsDone     = IsTimeOver || IsUnsafe;

    % Get reward
    if ~IsDone 
        Reward = 0;
    elseif  IsUnsafe
        Reward = 0;
    else
        Reward = 1; 
    end

end

end % EndOfFunction