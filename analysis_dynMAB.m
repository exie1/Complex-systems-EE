%%
clear p
p.a = 1.3; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
p.T = 9e2; % simulation time: integer multiple of MAB_steps pls
MAB_steps = 300;
num_parallel = 1;

payoffs = csvread('payoffs.csv')';
%% --------Defining stimuli and running simulation--------------------

R = pi/2;
theta = pi/2;   % for triangle stim
p.location = [-1,1;1,1;-1,-1;1,-1]*pi/2;
p.depth = [1,1,1,1];
p.sigma2 = [1,1,1,1]*0.32;
p.radius2 = [1,1,1,1].^2;
p.maxVal_d = 10;
% p.maxVal_s = 0.3;

p.rewardMu = payoffs(1,:);
p.rewardSig = zeros(1,4) + 4;
p.temp = 2;

tic
[X,t,history,history_rad] = fHMC_dynMABGaussian(p,payoffs,MAB_steps);
toc

optimal = sum(max(payoffs,[],2));
regret = 1 - (sum(history(2,:))/optimal);
[cnt_unique, uniq] = hist(history(1,:),unique(history(1,:)));
disp('Proportion of samples + overall regret')
disp([cnt_unique/sum(cnt_unique),regret])


% Plotting
figure
subplot(1,3,1)
plot(history(1,:))
xlabel('Step')
ylabel('Chosen option')
title('Choice history')
xlim([0,MAB_steps])


subplot(1,3,2)
histogram2(X(1,:,1),X(2,:,1),'Normalization','probability','FaceColor','flat')
xlabel('x')
ylabel('y')
title('Sampled distribution')

subplot(1,3,3)
for plt = 1:length(p.rewardMu)
    hold on
    plot(history_rad(plt,:))
end
xlabel('MAB step')
ylabel('Depth')
legend('Well 1', 'Well 2', 'Well 3', 'Well 4')
xlim([0,300])
title('Radius history')

%% Looping stuff
p.maxVal_d = 10;
maxval_list = linspace(0.5,2.5,20);
res_list = [];
for n = 1:length(maxval_list)
    p.temp = maxval_list(n);
    p.sigma2 = [1,1,1,1]*0.3;
    [X,t,history,history_rad] = fHMC_dynMABGaussian(p,payoffs,MAB_steps);
    regret = 1 - (sum(history(2,:))/optimal);
    res_list = [res_list, regret];
end


