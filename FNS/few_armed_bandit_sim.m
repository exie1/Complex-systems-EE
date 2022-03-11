a = 1.3; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e2; % simulation time
num_parallel = 30;

%--------Defining stimuli and running simulation--------------------
R = pi/2;
theta = pi/2;
p.location = [R*cos(theta),R*sin(theta); 
    R*cos(theta+2*pi/3),R*sin(theta+2*pi/3);
    R*cos(theta+4*pi/3),R*sin(theta+4*pi/3)];
p.depth = [1,1,1];
p.sigma2 = ([1,1,1]/2).^2;
ths = 2*sqrt(p.sigma2);


tic
[X,t] = fHMC_opt(T,a,p,num_parallel);
[mean_dur,total_dur] = closest_stim(X,p,ths,num_parallel);
rewards = gaussian_rewards(X,p);
toc

rewards_total = sum(rewards,1)/size(X,2);
disp([mean(rewards_total),std(rewards_total)])
disp([mean(mean_dur,2),std(mean_dur,0,2),mean(total_dur,2),std(total_dur,0,2)])

%% ---------------------Displaying results--------------------------
figure

subplot(1,3,1)
rewards = gaussian_rewards(X,p);
plot3(X(1,:,1),X(2,:,1),rewards(:,1),'.','markersize',1) % use regret/optimal score
xlabel('x')
ylabel('y')

subplot(1,3,2)
hold on
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

width_list = linspace(1,0.1,30);
avgs = 30; % average more or increase sampling time for more stimuli
total_times = [];
mean_times = [];
for i = 1:length(width_list)
    p.sigma2 = ([1,1,width_list(i)]/2).^2;
    ths = 2*sqrt(p.sigma2); 
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
plot(width_list,total_times(1,:))
plot(width_list,total_times(2,:))
plot(width_list,total_times(3,:))
plot(width_list,sum(total_times,1))
xlabel('Relative width of stimulus 3')
ylabel('total duration in stimuli (s)')
title('Total duration in stimuli over 100 s simulation')
legend('Stimulus 1', 'Stimulus 2','Stimulus 3 (varying width)','Sum')

