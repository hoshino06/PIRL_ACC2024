function [loss,gradients] = modelLoss(LossInput)

    % Deep Q-Learning loss.　(rl.grad.dqLoss)
    net = LossInput.net;
    obs = LossInput.obs;
    QPrediction = forward(net,obs);
    QPrediction = QPrediction(LossInput.ActionIdxMat);
    QPrediction = reshape(QPrediction,size(LossInput.Target));
    LossD = mse(QPrediction, LossInput.Target,'DataFormat','CB');
    
    % PDE loss
    xpde = LossInput.xpde;
    Qsa  = forward(net,xpde);
    [V, actIdx] = max(Qsa);
    act = LossInput.actInfo.getElementValue(actIdx);
    gradV = dlgradient(sum(V,"all"), xpde, EnableHigherDerivatives=true); 
    gradV2_x1 = dlgradient( sum(gradV(1,:),"all"), xpde, EnableHigherDerivatives=true);
    gradV2_x2 = dlgradient( sum(gradV(2,:),"all"), xpde, EnableHigherDerivatives=true);

    fun1  = @(x,u) -x(1)^3 - x(2);
    fun2  = @(x,u) x(1) +x(2) +u ;  
    convection = [ - arrayfun( @(i) fun1(xpde(:,i), act(i)), 1:LossInput.nBatch ); ...
                   - arrayfun( @(i) fun2(xpde(:,i), act(i)), 1:LossInput.nBatch ); ...
                   ones(1,LossInput.nBatch, "Like",xpde)  ]; 
    sigma = [0.2, 0.2];
    diffusion = sigma(1)^2*gradV2_x1(1,:) +sigma(2)^2*gradV2_x2(2,:);

    res = sum( gradV .* convection ) -(1/2)*diffusion; 

    zeroTarget = zeros(size(res),"like",res);
    LossP = mse(res,zeroTarget);    

    % Combine the losses. Use a weighted linear combination 
    % of loss (lossF and lossU) terms. 
    lambda = 1e-2; %1e-2; % 0.5*(LossD/LossP);  % Weighting factor
    loss = (1-lambda)*LossD + lambda*LossP;

    % disp(['LossD:' num2str(LossD,'%.2e'), ' LossP:' num2str(LossP,'%.2e')])

    % gradient
    gradients = dlgradient(loss, net.Learnables.Value );
    gradients = cellfun(@extractdata, gradients, 'UniformOutput', false);

end
