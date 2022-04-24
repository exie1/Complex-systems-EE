%% Problem setup
payoffs = csvread('payoffs.csv')';

clear p
R = pi/2;
theta = pi/2;   % for triangle stim
p.location = [-1,1;1,1;-1,-1;1,-1]*pi/2;
p.depth = [1,1,1,1];
p.sigma2 = [1,1,1,1]*0.3;
p.radius2 = [1,1,1,1].^2;
p.rewardMu = payoffs(1,:);
p.rewardSig = zeros(1,4) + 4;

p.dt = 1e-3; % integration time step
p.T = 9e2; % simulation time: integer multiple of MAB_steps pls
MAB_steps = 300;


%% --------Hyperparameters + simulating --------------------

% p.maxVal_s = 0.3;
p.a = 1.3; % Levy tail exponent
p.gam = 1; % strength of the Levy noise
p.beta = 0.5; % momentum term
p.temp = 0.5;
p.maxVal_d = 100;
p.l = 0.995;


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
subplot(3,1,1)
for plt = 1:length(p.rewardMu)
    hold on
    plot(payoffs(:,plt))
end
xlabel('Step')
ylabel('Mean payoff')
title('Mean payoff walk')
legend('Option 1','Option 2','Option 3','Option 4')

subplot(3,1,2)
plot(history(1,:),'.-')
xlabel('Step')
ylabel('Chosen option')
title('Choice history')
xlim([0,MAB_steps])


% subplot(1,3,2)
% histogram2(X(1,:,1),X(2,:,1),'Normalization','probability','FaceColor','flat')
% xlabel('x')
% ylabel('y')
% title('Sampled distribution')

subplot(3,1,3)
for plt = 1:length(p.rewardMu)
    hold on
    plot(history_rad(plt,:))
end
xlabel('MAB step')
ylabel('Depth')
legend('Well 1', 'Well 2', 'Well 3', 'Well 4')
xlim([0,300])
title('Depth history')

%% Looping stuff
% p.maxVal_d = 10;
% maxval_list = linspace(0.5,2.5,20);
res_list = [];

optimal = sum(max(payoffs,[],2));
tic
parfor n = 1:32 %length(maxval_list)
%     p.temp = maxval_list(n);
%     p.sigma2 = [1,1,1,1]*0.32;
%     p.depth = [1,1,1,1];
    [X,t,history,history_rad] = fHMC_dynMABGaussian(p,payoffs,MAB_steps);
    
    regret = 1 - (sum(history(2,:))/optimal);
    [cnt_unique, uniq] = hist(history(1,:),unique(history(1,:)));
    disp(['Proportion of samples + overall regret: ',num2str(n)])
    disp([cnt_unique/sum(cnt_unique),regret])
    res_list = [res_list, regret];
end
toc
%%
figure
hist(res_list);
xlabel('Regret')
disp(['Average regret: ', num2str(mean(res_list)),' Std: ', num2str(std(res_list))])


