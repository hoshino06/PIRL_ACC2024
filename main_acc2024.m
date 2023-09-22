% Q-learning for planer example 
clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
AgentOptions.DiscountFactor = 1.00;  %(default:0.99)
AgentOptions.MiniBatchSize  =   64; %(default:64)
AgentOptions.EpsilonGreedyExploration.EpsilonDecay = 0.0002; %(default:0.005)
AgentOptions.EpsilonGreedyExploration.EpsilonMin = 0.01; %(default:0.01)

%agent = rlDQNAgent(obsInfo,actInfo, AgentOptions);
agent = customDQNAgent(Critic, AgentOptions);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
trainOpts = rlTrainingOptions(...
    MaxEpisodes = 10000,...
    ScoreAveragingWindowLength = 100, ...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=1);

do_training = true;

trainOpts.Plots = "training-progress"; % "none"; % 

% if trainOpts.Plots == "training-progress"
%     trainOpts.UseParallel = true;
%     trainOpts.ParallelizationOptions.Mode = "async";
% end

if do_training == true
    trainResults = train(agent,env,trainOpts);
    save data/data.mat agent env trainResults

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementations of environment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [InitialObservation, LoggedSignal] = resetFunction()

    s  = sign( randn([2,1]) );
    r  = rand([2, 1]);

    X1  = s(1)*( r(1) * 1.5 );
    X2  = s(2)*( 0.8 + 0.2 * r(2) );  
    
    dt = 0.1;
    T  = randi([10, 30])*dt;

    LoggedSignal.State = [X1; X2; T];
    InitialObservation = LoggedSignal.State;

end

function [NextObs,Reward,IsDone,LoggedSignals] = stepFunction(Action,LoggedSignals)

    % Parameters
    dt = 0.1;
    sigma = 0.2;

    % Update state
    X     = LoggedSignals.State(1:2);
    U     = Action;

    fun = @(x,u) [ ( -x(1)^3 - x(2) ); ( x(1)+x(2)+ u ) ];
    [~,y] = ode45( @(t,x) fun(x,U), [0,dt], X);
    X_nxt = transpose( y(end,:) ) + sigma*sqrt(dt)*randn(2,1);

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
