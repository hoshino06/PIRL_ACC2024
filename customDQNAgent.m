classdef customDQNAgent < rl.agent.AbstractOffPolicyAgent
    % rlDQNAgent: Implements deep Q Network agent

    % Copyright 2017-2022 The MathWorks Inc.

    properties (Access = private)
        % update counter
        TargetUpdateCounter_ = 0

        % version indicator for backward compatibility
        Version = 3

        % Loss function
        LossFcn_
    end

    properties (Dependent, Access = private)
        % critic function
        Critic_
    end

    properties (Transient, Access = private)
        % Target critic function approximator
        TargetCritic_

        % Optimizer
        CriticOptimizer_
    end

    properties (Transient, Access = ?matlab.uitest.TestCase)
        CriticGradInfo_
    end


    %======================================================================
    % Public methods
    %======================================================================
    methods
        function this = customDQNAgent(critic,option)
            this = this@rl.agent.AbstractOffPolicyAgent(option);
            this = setCritic(this,critic);
            buildBuffer_(this);
        end

        % set/getCritic backward compatibility
        function this = setCritic(this,critic)
            % setCritic: Set the critic of the reinforcement learning agent
            % using the specified function object, CRITIC, which must be
            % consistent with the observations and actions of the agent.
            %
            %   AGENT = setCritic(AGENT,CRITIC)

            % take representation critic or function
            validateattributes(critic, {'rl.representation.rlQValueRepresentation','rl.function.AbstractQValueFunction'}, {'scalar', 'nonempty'}, mfilename, 'Critic', 2);

            if isa(critic,'rl.representation.rlAbstractRepresentation')
                % convert representation to function
                [critic,optimizerOptions] = representation2function(critic);
                this.AgentOptions.CriticOptimizerOptions = optimizerOptions;
            end
            critic = resetState(critic);
            critic = accelerate(critic,true);
            if rl.util.rlfeature('BuiltinGradientAcceleration') && ~hasState(critic)
                critic = accelerateBuiltin(critic,true);
            end
            this.validateActionInfo(critic.ActionInfo);

            % validate option and function (rnn)
            if ~isempty(this.ExplorationPolicy_)
                localCheckCriticPolicyDataSpec(critic,this.ExplorationPolicy_)
            end
            rl.agent.util.checkOptionFcnCompatibility(this.AgentOptions,critic);

            % validate if observation and action info is compatible with
            % existing policy
            if ~isempty(this.ExplorationPolicy_)
                rl.agent.util.checkObservationActionInfoCompatibility(this.ExplorationPolicy_,critic);
                this.Critic_ = critic;
            else
                this.ExplorationPolicy_ = rlEpsilonGreedyPolicy(critic);
                this.ExplorationPolicy_.ExplorationOptions = this.AgentOptions.EpsilonGreedyExploration;
                this.ExplorationPolicy_.SampleTime = this.SampleTime;
                this.ExplorationPolicy_.EnableEpsilonDecay = false;
            end

            % validate offline algorithm for RNN
            if hasState(critic)
                if isa(this.AgentOptions.BatchDataRegularizerOptions,'rl.option.rlConservativeQLearningOptions')
                    error(message('rl:agent:errorOfflineCQLRNNNotSupported'))
                end
            end

            % set target critic
            this.TargetCritic_ = critic;

            % create new optimizer
            this.CriticOptimizer_ = rlOptimizer(this.AgentOptions.CriticOptimizerOptions);

            % Select a loss function based on Q output type, RNN, and use
            % of offline training regularizer
            setCriticGradFcn_(this);            
        end
        function critic = getCritic(this)
            % getCritic: Return the critic function object, CRITIC, for the
            % specified reinforcement learning agent, AGENT.
            %
            %   CRITIC = getCritic(AGENT)

            critic = this.Critic_;
        end
    end

    % set/get
    methods
        function set.Critic_(this,critic)
            this.ExplorationPolicy_.QValueFunction = critic;
        end
        function critic = get.Critic_(this)
            critic = this.ExplorationPolicy_.QValueFunction;
        end
    end

    %======================================================================
    % Implementation of abstract methods for agent interface
    %======================================================================
    methods(Access = protected)
        function policy = getGreedyPolicy_(this)
            % return the greedy policy that always select action with the
            % highest state-action value Q(sma)
            policy = rlMaxQPolicy(this.Critic_);
        end

        function q0 = evaluateQ0_(this,observation)
            % estimate state value of the initial observation
            q0 = getMaxQValue(this.Critic_, observation);
        end

        function this = setLearnableParameters_(this,params)
            this.Critic_ = setLearnableParameters(this.Critic_,params.Critic);
        end
        function params = getLearnableParameters_(this)
            params.Critic = getLearnableParameters(this.Critic_);
        end

        function this = reset_(this)
            % reset buffer, policy and optimizer
            reset(this.ExperienceBuffer);
            reset(this.ExplorationPolicy_);
            this.CriticOptimizer_ = rlOptimizer(this.AgentOptions.CriticOptimizerOptions);
        end

        function setAgentOptions_(this,options)
            arguments
                this (1,1)
                options (1,1) rl.option.rlDQNAgentOptions
            end
            if ~isempty(this.AgentOptions)
                if ~isempty(this.ExplorationPolicy_)
                    % check compatibility of option and critic (e.g. rnn)
                    rl.agent.util.checkOptionFcnCompatibility(options,this.Critic_);
                    % update explore policy if noise option is changed
                    if ~isequal(options.EpsilonGreedyExploration,this.AgentOptions.EpsilonGreedyExploration)
                        this.ExplorationPolicy_ = rlEpsilonGreedyPolicy(this.Critic_);
                        this.ExplorationPolicy_.ExplorationOptions = options.EpsilonGreedyExploration;
                        this.ExplorationPolicy_.SampleTime = options.SampleTime;
                        this.ExplorationPolicy_.EnableEpsilonDecay = false;
                    end
                end

                % edit experience buffer if option changes
                if this.AgentOptions_.ExperienceBufferLength ~= options.ExperienceBufferLength
                    resize(this.ExperienceBuffer,options.ExperienceBufferLength);
                end

                % validate offline algorithm for RNN
                if hasState(this.Critic_)
                    if isa(options.BatchDataRegularizerOptions,'rl.option.rlConservativeQLearningOptions')
                        error(message('rl:agent:errorOfflineCQLRNNNotSupported'))
                    end
                end

                % recreate optimizer if option changes, only if critic
                % optimizer is already created
                % recreate optimizer if option changes and optimizer is already created
                oldOptimizerOptions = this.AgentOptions.CriticOptimizerOptions;
                newOptimizerOptions = options.CriticOptimizerOptions;
                this.CriticOptimizer_ = rl.agent.util.updateOptimizer(this.CriticOptimizer_,oldOptimizerOptions,newOptimizerOptions);
            end

            this.AgentOptions_ = options;
            this.SampleTime = options.SampleTime;

            % Select a loss function based on Q output type, RNN, and use
            % of offline training regularizer
            if ~isempty(this.ExplorationPolicy_)
                setCriticGradFcn_(this);
            end
        end

        function setCriticGradFcn_(this)
            this.LossFcn_ = lossFcnSelector_(this);
        end
        
        function setActorGradFcn_(this)
            % No op
        end
    end

    %======================================================================
    % Implementation of abstract methods for parallel mixin
    %======================================================================
    methods (Access = protected)
        function p = getWorkerPolicyParameters_(this)
            p = getLearnableParameters(this.Critic_);
        end
        function learnFromExperiencesInMemory_(this)
            % sample data from exp buffer and compute gradients
            [minibatch, maskIdx, sampleIdx, weights] = sample(this.ExperienceBuffer,...
                this.AgentOptions.MiniBatchSize,...
                SequenceLength = this.AgentOptions.SequenceLength,...
                DiscountFactor = this.AgentOptions.DiscountFactor,...
                NStepHorizon = this.AgentOptions.NumStepsToLookAhead);
            [this, learnData] = learnFromBatchData(this,minibatch,maskIdx,sampleIdx, weights);

            if isa(this.ExperienceBuffer,"rl.replay.rlPrioritizedReplayMemory")
                updateImportanceSamplingExponent(this.ExperienceBuffer);
            end

            % log the learning data
            learnData.Agent = this;
            rl.logging.internal.util.logAgentLearnData(learnData);
        end

        function [this, learnData] = learnFromBatchData_(this,minibatch,maskIdx, sampleIdx,weights)
            learnData.ActorLoss = single([]);
            learnData.CriticLoss = single([]);

            if ~isempty(minibatch)
                % enable epsilon decay in the policy
                this.ExplorationPolicy_.EnableEpsilonDecay = true;
                % perform the learning on the representation
                [criticGradient, criticLoss] = criticLearn_(this, minibatch, maskIdx, sampleIdx, weights);
                [this.Critic_,this.CriticOptimizer_] = update(this.CriticOptimizer_,...
                    this.Critic_,...
                    criticGradient);

                % update target critic
                % targetUpdateFrequency > 1 yields "periodic" updates
                % smoothingFactor < 1 yields "soft" updates
                % targetUpdateFrequency > 1 & smoothingFactor < 1 yields "periodic-soft" updates
                this.TargetUpdateCounter_ = this.TargetUpdateCounter_ + 1;
                if mod(this.TargetUpdateCounter_,this.AgentOptions.TargetUpdateFrequency) == 0
                    this.TargetCritic_ = syncParameters(this.TargetCritic_,this.Critic_,this.AgentOptions.TargetSmoothFactor);
                    this.TargetUpdateCounter_ = 0;
                end

                learnData.CriticLoss = criticLoss;
            end
        end
    end

    %======================================================================
    % Learning methods
    %======================================================================
    methods (Access = private)
        function [criticGradient, criticLoss] = criticLearn_(this, miniBatch, maskIdx, sampleIdx, weights)
            % update the critic against a minibatch set
            % MaskIdx is for DRQN. MaskIdx indicates actual inputs (true)
            % and padded inputs (false). MaskIdx is empty for DQN.
            % MaskIdx is a tensor of size [1 x MiniBatchSize x SequenceLength].

            % unpack experience
            doneIdx = (miniBatch.IsDone == 1);
            discount = this.AgentOptions.DiscountFactor ^ ...
                this.AgentOptions.NumStepsToLookAhead;

            % reset hidden state if network is RNN (LSTM)
            if hasState(this)
                % save the current hidden state to use at the next agent
                % step
                currentState = getState(this.Critic_);
                % reset hidden state before updating
                this.Critic_ = resetState(this.Critic_);
            end

            % compute target Q values
            targetQValues = getTargetQ_(this, miniBatch.NextObservation, miniBatch.Reward, doneIdx, discount);

            % build loss variable struct
            qType = getQType(this.Critic_);
            lossInput.Target = targetQValues;

            if hasState(this)
                % RNN uses a mask to ignore padded inputs.
                lossInput.MaskIdx = maskIdx;
            end

            % Compute gradient
            if qType == "multiOutput"
                [batchSize,sequenceLength] = inferDataDimension(this.ActionInfo, miniBatch.Action{1});
                actionIdxMat = getElementIndicationMatrix(this.ActionInfo,miniBatch.Action,...
                    batchSize*sequenceLength);
                lossInput.ActionIdxMat = reshape(actionIdxMat,[],batchSize, sequenceLength);

                % Additional input info for ConservativeQLearning
                if isa(this.AgentOptions.BatchDataRegularizerOptions,'rl.option.rlConservativeQLearningOptions')
                    lossInput.CQLAlpha = this.AgentOptions.BatchDataRegularizerOptions.MinQValueWeight;
                end

                if isa(this.ExperienceBuffer,"rl.replay.rlPrioritizedReplayMemory")
                    lossInput.Weights = reshape(weights, size(lossInput.Target));
                    [criticGradient, gradInfo] = gradient(this.Critic_,this.LossFcn_,...
                        miniBatch.Observation, lossInput);
                    Loss = gradInfo{1};
                    TDError = gradInfo{2};
                    updatePriorities(this.ExperienceBuffer, abs(TDError), sampleIdx);

                    %%%
                    disp('Caution! Modification not applied to Prioritized Experienced Replay: See line 319')
                else
    
                    %%%% Original gradient %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                    
                    %[criticGradient, gradInfo] = gradient(this.Critic_,this.LossFcn_,...
                    %    miniBatch.Observation, lossInput);
                    %disp('-----------------------------')
                    %disp(criticGradient{1}(1,:)) 

                    %%%% Custom gradient %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                    lossInput.net = this.Critic_.getModel;
                    lossInput.actInfo = this.ActionInfo;

                    % samples from experiences
                    obs = miniBatch.Observation{:};
                    obs = reshape(obs, size(obs,1), size(obs,3));
                    lossInput.obs  = dlarray(obs, 'CB');
                    
                    % samples for boundary (tau=0; inside safe set)
                    lossInput.nFin = 32;
                    x_min = [-1.5; -1.0;   0]; 
                    x_max = [ 1.5;  1.0;   0];
                    xfin = x_min + (x_max - x_min) .* ...
                            rand(prod(this.ObservationInfo.Dimension), lossInput.nFin);
                    lossInput.xfin = dlarray(xfin, 'CB');                                        

                    % samples for boundary (outside safe set)
                    lossInput.nUnsafe = 32;
                    x_min = [-1.5;  1.0;   0]; 
                    x_max = [ 1.5;  1.2;   3];
                    xunsafe = x_min + (x_max - x_min) .* ...
                                rand(prod(this.ObservationInfo.Dimension), lossInput.nUnsafe);                    
                    x2_sign = sign(randn([1,lossInput.nUnsafe]));
                    xunsafe(2,:) = xunsafe(2,:) .* x2_sign;
                    lossInput.xunsafe = dlarray(xunsafe, 'CB');                                       

                    % samples for pde loss (inside boundary)
                    lossInput.nPDE = 32;
                    x_min = [-1.5; -0.9;   0]; 
                    x_max = [ 1.5;  0.9; 2.5];
                    xpde = x_min + (x_max - x_min) .* ...
                            rand(prod(this.ObservationInfo.Dimension), lossInput.nPDE);
                    lossInput.xpde = dlarray(xpde, 'CB');

                    % custom loss function
                    [CustomLoss, CustomGrad]  = dlfeval(@customLoss, lossInput);
                    criticGradient = CustomGrad;

                    %disp(CustomGrad{1}(1,:)) 

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                end
            else
                disp('Caution! Modification not applied to Single Output Value function: See line 372')

                % Additional input info for ConservativeQLearning
                if isa(this.AgentOptions.BatchDataRegularizerOptions,'rl.option.rlConservativeQLearningOptions')
                    lossInput.CQLAlpha = this.AgentOptions.BatchDataRegularizerOptions.MinQValueWeight;
                    lossInput.MaxQ = getMaxQValue(this.Critic_, miniBatch.Observation);
                end

                if isa(this.ExperienceBuffer,"rl.replay.rlPrioritizedReplayMemory")
                    lossInput.Weights = reshape(weights, size(lossInput.Target));
                    [criticGradient, gradInfo] = gradient(this.Critic_,this.LossFcn_,...
                        [miniBatch.Observation,miniBatch.Action], lossInput);
                    Loss = gradInfo{1};
                    TDError = gradInfo{2};
                    updatePriorities(this.ExperienceBuffer, abs(TDError), sampleIdx);
                else
                    [criticGradient, gradInfo] = gradient(this.Critic_,this.LossFcn_,...
                        [miniBatch.Observation,miniBatch.Action], lossInput);
                end
            end

            if hasState(this)
                % after updating, recover the state using the saved hidden state
                this.Critic_ = setState(this.Critic_, currentState);
            end
            %this.CriticGradInfo_ = gradInfo;
            
            %%%% criticLoss
            % criticLoss = rl.logging.internal.util.extractLoss(gradInfo);
            criticLoss = CustomLoss;

        end

        function lossFcn = lossFcnSelector_(this)
            % lossFcnSelector_ selects a loss function based on Q type
            % (single or vector), type of replay memory, and the use of
            % offline training regularizer

            qType = getQType(this.Critic_);
            if hasState(this)
                % loss function for DRQN (DQN that uses LSTM).
                if isa(this.AgentOptions.BatchDataRegularizerOptions,'rl.option.rlConservativeQLearningOptions')
                    error(message('rl:agent:errorOfflineCQLRNNNotSupported'))
                else
                    lossFcn = @rl.grad.drqLoss;
                end
            else
                if qType == "multiOutput"
                    % loss function for DQN that not use RNN
                    if isa(this.AgentOptions.BatchDataRegularizerOptions,'rl.option.rlConservativeQLearningOptions')
                        if isa(this.ExperienceBuffer,"rl.replay.rlPrioritizedReplayMemory")
                            lossFcn = @rl.grad.dqCQLWeightedLoss;
                        else
                            lossFcn = @rl.grad.dqCQLLoss;
                        end
                    else
                        if isa(this.ExperienceBuffer,"rl.replay.rlPrioritizedReplayMemory")
                            lossFcn = @rl.grad.dqWeightedLoss;
                        else
                            lossFcn = @rl.grad.dqLoss;
                        end
                    end
                else
                    if isa(this.AgentOptions.BatchDataRegularizerOptions,'rl.option.rlConservativeQLearningOptions')
                        if isa(this.ExperienceBuffer,"rl.replay.rlPrioritizedReplayMemory")
                            lossFcn = @rl.grad.rlmseWeightedLossWithCQL;
                        else
                            lossFcn = @rl.grad.rlmseLossWithCQL;
                        end
                    else
                        if isa(this.ExperienceBuffer,"rl.replay.rlPrioritizedReplayMemory")
                            lossFcn = @rl.grad.rlmseWeightedLoss;
                        else
                            lossFcn = @rl.grad.rlmseLoss;
                        end
                    end
                end
            end
        end
    end

    %======================================================================
    % Helper methods
    %======================================================================
    methods (Access = private)
        function targetQValues = getTargetQ_(this, nextObservation, reward, doneIdx, discount)
            % compute Target Q values

            if this.AgentOptions.UseDoubleDQN
                % DoubleDQN: r + DiscountFactor*Qtarget[s',a' = argmax(qNetwork(s'))]
                % First we select a' using Critic.
                [tempTargetQValues, maxActionIndices]  = getMaxQValue(this.Critic_, nextObservation);

                % Secondly we evaluate target critic using s' and a'.
                if strcmpi(getQType(this.Critic_),'singleOutput')
                    [batchSize,sequenceLength] = inferDataDimension(this.ObservationInfo(1), nextObservation{1});
                    maxActions = getElementValue(this.ActionInfo, maxActionIndices);
                    maxActions = reshape(maxActions, [this.ActionInfo.Dimension batchSize sequenceLength]);
                    maxActions = rl.util.cellify(maxActions);
                    targetQValues = getValue(this.TargetCritic_, nextObservation, maxActions);
                else
                    % Get one hot encoding (logical array)
                    numActions = getNumberOfElements(this.ActionInfo);
                    maxActionIndicesOnehot = (maxActionIndices==(1:numActions)');
                    % Evaluate Q value
                    targetQValues = getValue(this.TargetCritic_, nextObservation);
                    targetQValues = targetQValues(maxActionIndicesOnehot);
                    targetQValues = reshape(targetQValues,size(tempTargetQValues));
                end
            else
                % DQN:       r + DiscountFactor*Qtarget[s',a' = argmax(targetNetwork(s'))]
                targetQValues = getMaxQValue(this.TargetCritic_, nextObservation);
            end

            targetQValues(~doneIdx) = reward(~doneIdx) + ...
                discount.*targetQValues(~doneIdx);

            % for terminal step, use the immediate reward (no more next state)
            targetQValues(doneIdx) = reward(doneIdx);
        end
    end

    methods (Hidden)
        function prop = qeGetProp(this,propName)
            prop = this.(propName);
        end
        function qeSetProp(this,propName,value)
            this.(propName) = value;
        end
        function targetQValues = qeGetTargetQ(this, nextObservation, reward, doneIdx, discount)
            targetQValues = getTargetQ_(this, nextObservation, reward, doneIdx, discount);
        end
        function setExplorationDecay(this,doDecay)
            this.ExplorationPolicy_.EnableEpsilonDecay = doDecay;
        end
        function lossFcn = qeLossFcnSelector(this)
            lossFcn = lossFcnSelector_(this);
        end
    end

    methods (Static, Hidden)
        function validateActionInfo(actionInfo)
            if ~isa(actionInfo,'rl.util.RLDataSpec')
                error(message('rl:agent:errInvalidActionSpecClass'))
            end

            % DQN not support continuous action data spec
            if rl.util.isaSpecType(actionInfo, 'continuous')
                error(message('rl:agent:errDQNContinuousActionSpec'))
            end

            % DQN not support actions that are matrix or tensor since
            % getElementIndicationMatrix in rlFiniteSetSpec does not
            % support them. It supports only scalar and vector.
            if iscell(actionInfo.Elements)
                singleElement = actionInfo.Elements{1};
            else
                singleElement = actionInfo.Elements(1);
            end
            if ~isscalar(singleElement) && ~isvector(singleElement)
                error(message('rl:agent:errDQNNotSupportMatrixTensorAction'))
            end
        end
    end

    %======================================================================
    % save/load
    %======================================================================
    methods
        function s = saveobj(this)
            % save user-facing objects to reconstruct in
            % loadobj
            s.AgentOptions = this.AgentOptions;
            s.Critic = this.Critic_;
            s.Version = this.Version;
            if this.AgentOptions.InfoToSave.ExperienceBuffer
                s.ExperienceBuffer = this.ExperienceBuffer;
            end
            if this.AgentOptions.InfoToSave.Optimizer
                s.CriticOptimizer = this.CriticOptimizer_;
            end
            if this.AgentOptions.InfoToSave.Target
                s.TargetCritic = this.TargetCritic_;
            end
            if this.AgentOptions.InfoToSave.PolicyState
                s.PolicyState = getState(this.ExplorationPolicy_);
            end
        end
    end
    methods (Static)
        function obj = loadobj(s)
            % reconstruct agent from saved data
            % Version 3 (current): starting R2022a
            % Version 2 (representation API): R2020a-R2021b
            % Version 1 (old representation API): R2019a-R2019b
            % Caveat:
            % - 19a, 19b, 20a dont have Version field
            if isstruct(s)
                % assign version fielf for versions that dont have this
                % field
                if ~isfield(s,'Version')
                    if isa(s.Critic,'rl.util.rlAbstractRepresentation')
                        % R2019a-R2019b, old representation API
                        s.Version = 1;
                    else
                        % R2020a, representation API
                        s.Version = 2;
                    end
                end

                switch s.Version
                    case 3
                        % Version 3 (current): starting R2022a
                        % create dqn agent from function object
                        % register any information stored in InfoToSave
                        obj = rlDQNAgent(s.Critic,s.AgentOptions);
                        if s.AgentOptions.InfoToSave.ExperienceBuffer
                            obj.ExperienceBuffer = s.ExperienceBuffer;
                        end
                        if s.AgentOptions.InfoToSave.Optimizer
                            obj.CriticOptimizer_ = s.CriticOptimizer;
                        end
                        if s.AgentOptions.InfoToSave.Target
                            obj.TargetCritic_ = s.TargetCritic;
                        end
                        if s.AgentOptions.InfoToSave.PolicyState
                            obj.ExplorationPolicy_ = setState(obj.ExplorationPolicy_,s.PolicyState);
                        end
                    case 2
                        % Version 2 (representation API): R2020a-R2021b
                        % saved agent use representation object
                        % Conversion from representation to function
                        % (e.g. rlQValueRepresentation to rlQValueFunction)
                        % is handled in informal API of agent
                        obj = rlDQNAgent(s.Critic,s.AgentOptions_);
                        obj.TargetCritic_ = representation2function(s.TargetCritic);

                        if obj.AgentOptions_.SaveExperienceBufferWithAgent
                            % only load the experience buffer if
                            % SaveExperienceBufferWithAgent is true
                            collectExperienceFromOldFormat(obj.ExperienceBuffer,s.ExperienceBuffer);
                        end
                    case 1
                        % Version 1 (old representation API): R2019a-R2019b
                        % convert rl.util.rlAbstractRepresentation to new
                        % function class
                        criticModel = s.Critic.getModel;
                        targetCriticModel = s.TargetCritic.getModel;
                        s.AgentOptions_.CriticOptimizerOptions = rl.internal.dataTransformation.repOptions2OptimOptions(s.Critic.Options);
                        useDevice = s.Critic.Options.UseDevice;
                        if isa(criticModel,'DAGNetwork') || isa(criticModel,'nnet.cnn.LayerGraph')
                            s.Critic = rlQValueFunction(criticModel,...
                                s.ObservationInfo, s.ActionInfo, ...
                                "ObservationInputNames",string(s.Critic.ObservationNames), ...
                                "ActionInputNames",string(s.Critic.ActionNames));
                            s.TargetCritic = rlQValueFunction(targetCriticModel,...
                                s.ObservationInfo, s.ActionInfo, ...
                                "ObservationInputNames",string(s.TargetCritic.ObservationNames), ...
                                "ActionInputNames",string(s.TargetCritic.ActionNames));
                        else
                            s.Critic = rlQValueFunction(criticModel,...
                                s.ObservationInfo, s.ActionInfo);
                            s.TargetCritic = rlQValueFunction(targetCriticModel,...
                                s.ObservationInfo, s.ActionInfo);
                        end
                        s.Critic.UseDevice = useDevice;
                        s.TargetCritic.UseDevice = useDevice;

                        obj = rlDQNAgent(s.Critic,s.AgentOptions_);
                        obj.TargetCritic_ = s.TargetCritic;
                        if obj.AgentOptions_.SaveExperienceBufferWithAgent
                            % only load the experience buffer if
                            % SaveExperienceBufferWithAgent is true
                            collectExperienceFromOldFormat(obj.ExperienceBuffer,s.ExperienceBuffer);
                        end
                end
            else
                % this should not be hit
                obj = s;
            end
        end
    end
end

%==========================================================================
% Local functions
%==========================================================================
function localCheckCriticPolicyDataSpec(critic,policy)
if ~hasState(critic) && hasState(policy.QValueFunction)
    % if the original critic network is an RNN
    % but the new critic is not an RNN, suggest create
    % new agent with SequenceLength = 1
    error(message('rl:agent:errCriticChangedtoStateless'))
elseif hasState(critic) && ~hasState(policy.QValueFunction)
    % if the original critic network is not an RNN
    % but the new critic is an RNN, suggest create
    % new agent with SequenceLength > 1
    error(message('rl:agent:errCriticChangedtoRNN'))
end
end
