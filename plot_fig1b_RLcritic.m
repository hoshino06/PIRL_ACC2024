% Evaluation of Leared Safety Probability
clear

T = 2.0; % horizon of PSC

plot_on = true;

%% Read learned agent
load("data/data_lambda_2e-2.mat","agent")  % 2e-2

QFunc = agent.getCritic;

%% Show learned safe probability (wrt Del/V0)
n_rez = 100;
Xlim = [-1.5, 1.5]; 
Ylim = [-1.2, 1.2]; 
xmin = Xlim(1); xmax = Xlim(2); dx = (xmax-xmin)/(n_rez-1);
ymin = Ylim(1); ymax = Ylim(2); dy = (ymax-ymin)/(n_rez-1);
[X,Y] = meshgrid(xmin:dx:xmax, ymin:dy:ymax); % indeces for x and y axis

SafeP  = zeros(n_rez,n_rez);
Action = zeros(n_rez,n_rez);
for i = 1:n_rez
   for j = 1:n_rez

    state = [X(i,j) Y(i,j) T];
    SafeP(i,j) = QFunc.getMaxQValue({state}); 
    
    act = agent.getAction(state');
    Action(i,j) = act{:};

   end
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
xlabel('x_1')
ylabel('x_2')

% vector feild 
n_rez = 15; 
Xlim = [-1.5, 1.5]; 
Ylim = [-1.2, 1.2]; 
xmin = Xlim(1); xmax = Xlim(2); dx = (xmax-xmin)/(n_rez-1);
ymin = Ylim(1); ymax = Ylim(2); dy = (ymax-ymin)/(n_rez-1);
[X,Y] = meshgrid(xmin:dx:xmax, ymin:dy:ymax); % indeces for x and y axis

U = zeros(size(X));
for i = 1:n_rez
    for j = 1:n_rez
        Ucell  = agent.getAction([X(i,j); Y(i,j); T]);
        U(i,j) = Ucell{1};
    end
end

DX = -X^3 - Y;
DY = X + Y + U;
quiver3(X,Y,ones(n_rez)*1.2, DX,DY, zeros(n_rez), 'r', LineWidth=1.2);

hold off

fontsize(20,"points")
%set(gca, 'LooseInset', get(gca, 'TightInset'));
set(gca, 'LooseInset', [0.13, 0.11, 0.0, 0.075]);

if plot_on
    saveas(gcf, "data/2d_RL_critic.eps",'epsc')
end


%%% Controller  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2)
n_rez = 100;
Xlim = [-1.5, 1.5]; 
Ylim = [-1.2, 1.2]; 
xmin = Xlim(1); xmax = Xlim(2); dx = (xmax-xmin)/(n_rez-1);
ymin = Ylim(1); ymax = Ylim(2); dy = (ymax-ymin)/(n_rez-1);
[X,Y] = meshgrid(xmin:dx:xmax, ymin:dy:ymax); % indeces for x and y axis

surf(X,Y,Action,'EdgeColor','none')%, ...
        %"ShowText",true)
colorbar
view(2)
xlim(Xlim)
ylim(Ylim)
xlabel('x_1')
ylabel('x_2')

