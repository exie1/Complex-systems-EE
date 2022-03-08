a = 1.3; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e2; % simulation time
num_parallel = 2;

%--------Defining stimuli and running simulation--------------------
p.location = [-pi/2,pi/2; pi/2,-pi/2];
p.depth = [1,1];
p.sigma2 = ([1,1]/2).^2;
ths = 2*sqrt(p.sigma2);


tic
[X,t] = fHMC_opt(T,a,p,num_parallel);
[mean_dur,total_dur] = closest_stim(X,p,ths,num_parallel);
rewards = gaussian_rewards(X,p);
toc

%% ---------------------Displaying results--------------------------
figure
subplot(1,2,1)
rewards = gaussian_rewards(X,p);
plot3(X(1,:,1),X(2,:,1),rewards(:,1),'.','markersize',1) % use regret/optimal score
rewards_total = sum(rewards,1)/size(X,2);
disp([mean(rewards_total),std(rewards_total)])

subplot(1,2,2)
plot(X(1,:,1),X(2,:,1),'.','markersize',1)
hold on
viscircles(p.location, ths);
disp([mean(mean_dur,2),std(mean_dur,0,2),mean(total_dur,2),std(total_dur,0,2)])

%% Looping business 


depth_list = linspace(1,0.1,30);
avgs = 30;
total_times = [];
mean_times = [];
for i = 1:length(depth_list)
    p.depth = ([depth_list(i),1]/2).^2;
    [X,t] = fHMC_opt(T,a,p,avgs);
    [mean_dur,total_dur] = closest_stim(X,p,ths,avgs);
    total_times = [total_times, mean(total_dur,2)];
    mean_times = [mean_times, mean(mean_dur,2)];
    disp([mean(mean_dur,2),mean(total_dur,2)])
    disp(i)
end

%%
figure 
hold on
plot(depth_list,total_times(1,:))
plot(depth_list,total_times(2,:))
plot(depth_list,sum(total_times,1))
xlabel('Relative strength of stimulus 1')
ylabel('total duration in stimuli (s)')
title('Total duration in stimuli over 100 s simulation')
legend('Stimulus 1 (varying strength)', 'Stimulus 2','Sum')

