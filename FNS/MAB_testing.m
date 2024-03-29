clear p
a = 1.3; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e3; % simulation time
MAB_steps = 500;
window = T/p.dt/MAB_steps;
num_parallel = 1;

%--------Defining stimuli and running simulation--------------------

R = pi/2;
theta = pi/2;   % for triangle stim
p.location = [R*cos(theta),R*sin(theta); 
    R*cos(theta+2*pi/3),R*sin(theta+2*pi/3);
    R*cos(theta+4*pi/3),R*sin(theta+4*pi/3)];
p.depth = [1,1,1];

p.sigma2 = [1,1,1].^2;
% p.rewardMu = [3,4,5];       % should initialise this as sampled values
% p.rewardSig = [5,4,3]/2;
p.rewardMu = 2 + 3*(1:3);
p.rewardSig = 0.5*p.rewardMu;

tic
[X,t,history,history_rad] = fHMC_MAB(T,a,p,window);
toc

optimal = max(p.rewardMu)*MAB_steps;
regret = 1 - (sum(history(2,:))/optimal);
[cnt_unique, uniq] = hist(history(1,:),unique(history(1,:)));
disp('Number of times each option is sampled + overall regret')
disp([cnt_unique,regret])
%%
figure
subplot(2,3,1)
plot(history(1,:))
xlabel('Step')
ylabel('Chosen option')
title('Choice history')
xlim([0,MAB_steps])

subplot(2,3,2)
histogram2(X(1,:,1),X(2,:,1),'Normalization','probability','FaceColor','flat')
xlabel('x')
ylabel('y')
title('Sampled distribution')

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


%% Other strategies

numTries = 1e5;
pureEE_lst = zeros(2,numTries);
for testing = 1:numTries
    reward_i = p.rewardSig.*randn(1,3)+p.rewardMu;
    [~,chosen1] = max(reward_i);
    [~,chosen2] = max(rand(1,3));
    pureEE_lst(1,testing) = p.rewardMu(chosen1)*MAB_steps;
    pureEE_lst(2,testing) = p.rewardMu(chosen2)*MAB_steps;
end
disp("Pure exploitation and pure exploration mean regret:")
disp([1 - (mean(pureEE_lst(1,:)))/optimal, ...
    1 - (mean(pureEE_lst(2,:)))/optimal])


%% Looping stuff

reco = [];
for es = 1:20
    p.rewardMu = 2 + 3*(1:3);
    p.rewardSig = 0.5*p.rewardMu;

    [X,t,history,history_rad] = fHMC_MAB(T,a,p,window,1);

    optimal = max(p.rewardMu)*MAB_steps;
    regret = 1 - (sum(history(2,:))/optimal);
    [cnt_unique, uniq] = hist(history(1,:),unique(history(1,:)));
    disp('Number of times each option is sampled + overall regret')
    disp([cnt_unique,regret])
    reco = [reco, regret];
end
  
