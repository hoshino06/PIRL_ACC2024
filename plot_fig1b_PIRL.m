% Evaluation of Leared Safety Probability
clear

T = 2.0; % horizon of PSC

MC_calc = false;
save_fig = true;

if MC_calc == true

    % Read learned agent
    data  = load("data/TD20Lam1e-2_S5000_B09/data4.mat","agent");
    agent = data.agent;

    % Show learned safe probability (wrt Del/V0)
    n_rez = 100;
    Xlim = [-1.5, 1.5]; 
    Ylim = [-1.2, 1.2]; 
    xmin = Xlim(1); xmax = Xlim(2); dx = (xmax-xmin)/(n_rez-1);
    ymin = Ylim(1); ymax = Ylim(2); dy = (ymax-ymin)/(n_rez-1);
    [X,Y] = meshgrid(xmin:dx:xmax, ymin:dy:ymax); % indeces for x and y axis

    tic
    n_sample = 100;
    SafeP  = zeros(n_rez,n_rez);
    Action = zeros(n_rez,n_rez);
    parfor i = 1:n_rez
       for j = 1:n_rez
        SafeP(i,j) = safe_probability_MC(agent, X(i,j), Y(i,j), T, n_sample)
        state = [X(i,j) Y(i,j) T];
        act = agent.getAction(state');
        Action(i,j) = act{:};
    
       end
    end
    toc

    save data/data_fig1b_PIRL.mat n_rez Xlim Ylim X Y SafeP Action agent

else

    load("data/data_fig1b_PIRL.mat","n_rez","X","Xlim","Ylim","Y","SafeP","Action","agent")

end

%%%% Figure 1(b) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = figure(1); 

clf(f)
set(gcf, 'PaperPosition', [0 0 6 5]);

hold on

% safe probability
surf(X,Y,SafeP,'EdgeColor','none')%, ...
        %"ShowText",true)
colorbar('eastoutside','Limits',[0,1])
view(2)
xlim(Xlim)
ylim(Ylim)
%zlim([0,1])
xlabel('x1')
ylabel('x2')

% vector feild 
n_vec = 15; 
Xlim = [-1.5, 1.5]; 
Ylim = [-1.2, 1.2]; 
xmin = Xlim(1); xmax = Xlim(2); dx = (xmax-xmin)/(n_vec-1);
ymin = Ylim(1); ymax = Ylim(2); dy = (ymax-ymin)/(n_vec-1);
[Xvec,Yvec] = meshgrid(xmin:dx:xmax, ymin:dy:ymax); % indeces for x and y axis

U = zeros(size(Xvec));
parfor i = 1:n_vec
    for j = 1:n_vec
        Ucell  = agent.getAction([Xvec(i,j); Yvec(i,j); T]);
        U(i,j) = Ucell{1};
    end
end

DX = -Xvec^3 - Yvec;
DY = Xvec + Yvec + U;
quiver3(Xvec,Yvec,ones(n_vec)*1.2, DX,DY, zeros(n_vec), 'r', LineWidth=1.2);

hold off

fontsize(20,"points")
%set(gca, 'LooseInset', get(gca, 'TightInset'));
set(gca, 'LooseInset', [0.13, 0.11, 0.05, 0.075]);

if save_fig
    saveas(gcf, "data/fig1b_PIRL.eps",'epsc')
end


%%% Controller  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2)

surf(X,Y,Action,'EdgeColor','none')%, ...
        %"ShowText",true)
colorbar
view(2)
xlim(Xlim)
ylim(Ylim)
xlabel('x1')
ylabel('x2')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob = safe_probability_MC(agent, x1, x2, T, n_sample)

dt = 0.1;
sigma = 0.2;

% Monte Carlo simulation
samples  = ones(1, n_sample);

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
    prob = sum(samples)/n_sample;
end

end