% Evaluate learning process 
clear

res1 = load("data/data_lambda_0.mat","trainResults"); 
res2 = load("data/data_lambda_2e-2.mat","trainResults"); 
res3 = load("data/data_lambda_5e-2.mat","trainResults"); 


f = figure(1); clf(f);
f.Position = [800 400 650 250];

hold on
plot(res1.trainResults.EpisodeIndex(10:end), res1.trainResults.AverageReward(10:end))
plot(res2.trainResults.EpisodeIndex(10:end), res2.trainResults.AverageReward(10:end))
plot(res3.trainResults.EpisodeIndex(10:end), res3.trainResults.AverageReward(10:end))

hold off

legend('\lambda = 0','\lambda = 2e-2','\lambda = 5e-2', Location='best')
xlabel('Episodes')
ylabel('Average Return')

xlim([0,5000])
fontsize(11,"points")

saveas(gcf,"data/learning_process.eps",'epsc')


