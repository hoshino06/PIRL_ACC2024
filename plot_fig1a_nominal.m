clear

MC_calc = false;

T_interval = 2.0; % for T in parfor loop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System equation
t = sym('t');
x = sym('x',[1 2]).';
f = [ ( -x(1)^3 - x(2) ); ( x(1)+x(2) ) ];
g = [ 0; 1];
% Initial control law derived by feedback linearization
z = [ x(1) ; ( -x(1)^3 +x(2) )];
v_init = 0.4142*z(1) -1.3522*z(2);
u_init = max( min( 3*x(1)^5 + 3*x(1)^2*x(2) -x(2) + v_init, 1 ), -1);
fun = matlabFunction(f+g*u_init,"Vars", {[x(1); x(2)]});

% Monte Carlo simulation
n_rez = 100; 
Xlim = [-1.5, 1.5]; 
Ylim = [-1.2, 1.2]; 
xmin = Xlim(1); xmax = Xlim(2); dx = (xmax-xmin)/(n_rez-1);
ymin = Ylim(1); ymax = Ylim(2); dy = (ymax-ymin)/(n_rez-1);
[X,Y] = meshgrid(xmin:dx:xmax, ymin:dy:ymax); % indeces for x and y axis
SafeP  = zeros(n_rez,n_rez);

if MC_calc == true

    tic
    parfor i = 1:n_rez
       for j = 1:n_rez
    
           n_sample = 100;
           samples  = ones(1, n_sample);
           for n = 1:n_sample
    
               dt = 0.1;
    
               horizon = T_interval/dt; 
               x = [X(i,j); Y(i,j)];
    
               for k = 1:horizon
    
                   % is unsafe? 
                   if abs( x(2) ) > 1
                       samples(n)   = 0;
                       break
                   end
                   
                   % calculate next state
                   [~,y] = ode45( @(t,x) fun(x), [0,dt], x);
                   x = transpose( y(end,:) ) + 0.2*sqrt(dt)*randn(2,1);
    
               end
               % is unsafe? (check the final state)
               if abs( x(2) ) > 1
                   samples(n)   = 0;
               end
    
           end
           SafeP(i,j) = sum(samples)/n_sample;
    
       end
    end
    toc

    save data/data_fig1a_nominal.mat X Y SafeP

else

    load("data/data_fig1a_nominal.mat","X","Y","SafeP")  % 2e-2

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of safe probability
f = figure(1); 

clf(f)
set(gcf, 'PaperPosition', [0 0 6 5]);


surf(X,Y,SafeP,'EdgeColor','none')%, ...
        %"ShowText",true)
colorbar
hold on
view(2)
xlim(Xlim)
ylim(Ylim)
xlabel('x1')
ylabel('x2')


% Plot of vector feild 
n_rez = 15; 
Xlim = [-1.5, 1.5]; 
Ylim = [-1.2, 1.2]; 
xmin = Xlim(1); xmax = Xlim(2); dx = (xmax-xmin)/(n_rez-1);
ymin = Ylim(1); ymax = Ylim(2); dy = (ymax-ymin)/(n_rez-1);
[X,Y] = meshgrid(xmin:dx:xmax, ymin:dy:ymax); % indeces for x and y axis

V = 0.4142*X -1.3522*(-X^3 +Y);
U = max( min( 3*X^5 + 3*X^2*Y -Y + V, 1 ), -1);
DX = -X^3 - Y;
DY = X + Y + U;
quiver3(X,Y,ones(n_rez)*1.2, DX,DY, zeros(n_rez), 'r', LineWidth=1.2);
hold off

fontsize(20,"points")
%set(gca, 'LooseInset', get(gca, 'TightInset'));
set(gca, 'LooseInset', [0.13, 0.11, 0.05, 0.075]);

saveas(gcf,"data/fig1a_nominal.eps", 'epsc')
