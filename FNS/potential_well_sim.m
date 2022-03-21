a = 1.3; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e2; % simulation time
num_parallel = 20;

%--------Defining stimuli and running simulation--------------------


p.location = [-1,1; 1, -1]*pi/2; 
p.depth = 1 + zeros(length(p.location),1);
p.radius2 = 1^2 + zeros(length(p.location),1);
ths = sqrt(p.radius2);

tic
[X,t] = fHMC(T,a,p,num_parallel);
[mean_dur,total_dur] = closest_stim(X,p,ths,num_parallel);
rewards = quadratic_rewards(X,p);
toc

rewards_total = sum(rewards,1)/size(X,2);
disp([mean(rewards_total),std(rewards_total)])
disp([mean(mean_dur,2),std(mean_dur,0,2),mean(total_dur,2),std(total_dur,0,2)])

%% ---------------------Displaying results--------------------------
figure


subplot(1,3,1)
plot3(X(1,:,1),X(2,:,1),rewards(:,1),'.','markersize',1) 
xlabel('x')
ylabel('y')

subplot(1,3,2)
hist3(squeeze(X(:,:,1))',[50,50],'CDataMode','auto','FaceColor','interp')
% plot3(X(1,:,1),X(2,:,1),rewards(:,1)*1400,'.','markersize',1) 
xlabel('x')
ylabel('y')

subplot(1,3,3)
plot(X(1,:,1),X(2,:,1),'.','markersize',1)
hold on
viscircles(p.location, ths);
xlim([-pi,pi])
ylim([-pi,pi])
xlabel('x')
ylabel('y')

%% Looping business

avgs = 30; % should average more or increase sampling time for more stimuli
depth_list = linspace(1,10,20).^2;

total_times = [];
mean_times = [];
for i = 1:length(depth_list)
    p.depth(1) = depth_list(i);
    ths = sqrt(p.radius2); 
    [X,t] = fHMC(T,a,p,avgs);
    [mean_dur,total_dur] = closest_stim(X,p,ths,avgs);
    total_times = [total_times, mean(total_dur,2)];
    mean_times = [mean_times, mean(mean_dur,2)];
    disp([mean(mean_dur,2),mean(total_dur,2)])
    disp(i)
end


%%

figure 
hold on

plot(sqrt(depth_list),total_times(1,:))
plot(sqrt(depth_list),total_times(2,:))
plot(sqrt(depth_list),sum(total_times,1))
xlabel('depth of stimuli 1')
ylabel('total duration in stimuli (s)')
ylim([0,100])
title('Total duration in stimuli over 100 s simulation')
legend('Stimulus 1 (varying depth)', 'Stimulus 2','Sum')

