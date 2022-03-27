a = 1.3; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e3; % simulation time
MAB_steps = 500;
window = T/p.dt/MAB_steps;
num_parallel = 30;

%--------Defining stimuli and running simulation--------------------

R = pi/2;
theta = pi/2;   % for triangle stim
p.location = [R*cos(theta),R*sin(theta); 
    R*cos(theta+2*pi/3),R*sin(theta+2*pi/3);
    R*cos(theta+4*pi/3),R*sin(theta+4*pi/3)];
p.depth = [1,1,1];

p.radius2 = [1,1,1].^2;
p.rewardMu = [3,4,5];       % should initialise this as sampled values
p.rewardSig = [5,4,3]/2;

tic
<<<<<<< HEAD
[X,t,history_rewards,history_choices,history_rad] = fHMC_MAB(T,a,p,window,num_parallel);
toc
%%
optimal = max(p.rewardMu)*MAB_steps;
regret = mean(1 - (sum(history_rewards,2)/optimal));
[cnt_unique, uniq] = hist(history_choices',unique(history_choices));
disp('Mean number of times each option is sampled + mean regret')
disp([mean(cnt_unique,2)',regret])
=======
[X,t,history] = fHMC_MAB(T,a,p,window,num_parallel);
toc

regret = 1 - (sum(history(2,:))/max(p.rewardMu)*MAB_steps);
>>>>>>> parent of 146b0c1 (Fixed MAB simulation)
%%
trial = 2;

figure
<<<<<<< HEAD
subplot(2,3,1)
plot(history_choices(trial,:))
xlabel('Step')
ylabel('Chosen option')

subplot(2,3,2)
histogram2(X(1,:,trial),X(2,:,trial),'Normalization','probability','FaceColor','flat')
xlabel('x')
ylabel('y')

subplot(2,3,3)
for plt = 1:3
    hold on
    plot(squeeze(history_rad(trial,plt,:)))
end
xlabel('MAB step')
ylabel('Radius')

pdf_range = linspace(min(history(2,:)),max(history(2,:)),200);
disp('Mean and std of simulated payoff distribution')
for plt = 1:3
    subplot(2,3,3+plt)
    oneOpt = history_rewards(trial,:).*(history_choices(trial,:) == plt);  % Payoffs from option __ 
    hold on
    histogram(oneOpt(oneOpt~=0),'Normalization','probability')
    plot(pdf_range,normpdf(pdf_range,p.rewardMu(plt),p.rewardSig(plt)),'lineWidth',2)
    xlabel('Payoff')
    ylabel('Probability')
    disp([mean(oneOpt(oneOpt~=0)),std(oneOpt(oneOpt~=0))])
end


%% Find total duration?
=======
subplot(1,2,1)
plot(history(1,:))
xlabel('Step')
ylabel('Chosen option')

subplot(1,2,2)
histogram2(X(1,:,1),X(2,:,1),'Normalization','probability','FaceColor','flat')
xlabel('x')
ylabel('y')

%%
>>>>>>> parent of 146b0c1 (Fixed MAB simulation)

figure
oneOpt = history(2,:) .* (history(1,:) == 3);
histogram(oneOpt)
(5*500 - sum(history(2,:)))/(5*500)
