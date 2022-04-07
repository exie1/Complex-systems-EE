clear p
a = 1.3; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
p.tau = 1; % softmax sensitivity
T = 1e3; % simulation time
MAB_steps = 500;
window = T/p.dt/MAB_steps;
num_parallel = 3;

%--------Defining stimuli and running simulation--------------------

R = pi/2;
theta = pi/2;   % for triangle stim
p.location = [R*cos(theta),R*sin(theta); 
    R*cos(theta+2*pi/3),R*sin(theta+2*pi/3);
    R*cos(theta+4*pi/3),R*sin(theta+4*pi/3)];
p.depth = [1,1,1];

p.radius2 = [1,1,1].^2;
% p.rewardMu = [3,4,5];       % should initialise this as sampled values
% p.rewardSig = [5,4,3]/2;
p.rewardMu = 2 + 3*(1:3);
p.rewardSig = 0.5*p.rewardMu;

tic
[X,t,history_rewards,history_choices] = fHMC_MAB_avgd(T,a,p,window,num_parallel);
toc

optimal = max(p.rewardMu)*MAB_steps;
regret = 1 - (sum(history_rewards,2)/optimal)
%%
trial =43;
[cnt_unique, uniq] = hist(history_choices(trial,:),unique(history_choices(trial,:)));
disp(cnt_unique)
% disp('Number of times each option is sampled + overall regret')
% disp([cnt_unique,regret])

%%
trial = 1;

figure
subplot(1,2,1)
plot(history_choices(trial,:))
xlabel('Step')
ylabel('Chosen option')
title('Choice history')
xlim([0,MAB_steps])

subplot(1,2,2)
histogram2(X(1,:,trial),X(2,:,trial),'Normalization','probability','FaceColor','flat')
xlabel('x')
ylabel('y')
title('Sampled distribution')

%% Plotting

subplot(2,3,3)
for plt = 1:3
    hold on
    plot(history_rad(plt,:))
end
xlabel('MAB step')
ylabel('Radius')
legend('Well 1', 'Well 2', 'Well 3')
title('Radius history')

pdf_range = linspace(min(history(2,:)),max(history(2,:)),200);
disp('Mean and std of simulated payoff distribution')
for plt = 1:3
    subplot(2,3,3+plt)
    oneOpt = history(2,:).*(history(1,:) == plt);  % Payoffs from option __ 
    hold on
    histogram(oneOpt(oneOpt~=0),'Normalization','probability')
    plot(pdf_range,normpdf(pdf_range,p.rewardMu(plt),p.rewardSig(plt)),'lineWidth',2)
    xlabel('Payoff')
    ylabel('Probability')
    title(['Well ',num2str(plt),' payoff, sampled ' ,num2str(cnt_unique(plt)),' times'])
    disp([mean(oneOpt(oneOpt~=0)),std(oneOpt(oneOpt~=0))])
end