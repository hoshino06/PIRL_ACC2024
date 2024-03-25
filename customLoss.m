function [loss,gradients] = customLoss(LossInput)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % DQN loss term (copied from rl.grad.dqLoss)
    net = LossInput.net;
    obs = LossInput.obs;
    QPrediction = forward(net,obs);
    QPrediction = QPrediction(LossInput.ActionIdxMat);
    QPrediction = reshape(QPrediction,size(LossInput.Target));
    LossD = mse(QPrediction, LossInput.Target,'DataFormat','CB');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PDE loss (boundary condition; tau=0; safe region)
    xfin = LossInput.xfin;
    Qsa  = forward(net,xfin);
    V  = max(Qsa);
    target = ones(size(V),"like",V);
    LossPfin = mse(V, target);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PDE loss (boundary condition; unsafe region)
    xunsafe = LossInput.xunsafe;
    Qsa  = forward(net,xunsafe);
    V  = max(Qsa);
    target = zeros(size(V),"like",V);
    LossPunsafe = mse(V, target);    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PDE loss
    xpde = LossInput.xpde;
    Qsa  = forward(net,xpde);
    [V, actIdx] = max(Qsa);
    act = LossInput.actInfo.getElementValue(actIdx);
    gradV = dlgradient(sum(V,"all"), xpde, EnableHigherDerivatives=true); 

    fun1  = @(x,u) -x(1)^3 - x(2);
    fun2  = @(x,u) x(1) +x(2) + u ;  
    convection = [ - arrayfun( @(i) fun1(xpde(:,i), act(i)), 1:LossInput.nPDE ); ...
                   - arrayfun( @(i) fun2(xpde(:,i), act(i)), 1:LossInput.nPDE ); ...
                   ones(1,LossInput.nPDE, "Like",xpde)  ]; 
    
    gradV2_x1 = dlgradient( sum(gradV(1,:),"all"), xpde, EnableHigherDerivatives=true);
    gradV2_x2 = dlgradient( sum(gradV(2,:),"all"), xpde, EnableHigherDerivatives=true);

    sigma = [0.2, 0.2];
    diffusion = sigma(1)^2*gradV2_x1(1,:) +sigma(2)^2*gradV2_x2(2,:);

    res = sum( gradV .* convection ) -(1/2)*diffusion; 

    zeroTarget = zeros(size(res),"like",res);
    LossP = mse(res,zeroTarget);    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Combined loss 
    lambda = 1e-4;
    loss = LossD + LossPfin + LossPunsafe + lambda* LossP; 

    % disp(['LossD:' num2str(LossD,'%.2e'), ' LossP:' num2str(LossP,'%.2e')])

    % gradient
    gradients = dlgradient(loss, net.Learnables.Value );
    gradients = cellfun(@extractdata, gradients, 'UniformOutput', false);

end
