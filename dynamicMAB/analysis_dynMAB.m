%% Problem setup
payoffs = csvread('payoffs\payoffs_step2.csv')';

clear p

% p.location = [-1,1;1,1;-1,-1;1,-1]*pi/2;
R = 1.5;
theta = pi/2;   % for triangle stim
p.location = [R*cos(theta),R*sin(theta); 
    R*cos(theta+2*pi/3),R*sin(theta+2*pi/3);
    R*cos(theta+4*pi/3),R*sin(theta+4*pi/3)];
p.depth = [1,1,1];


p.rewardMu = payoffs(1,:);
p.rewardSig = zeros(1,3) + 4;

p.dt = 1e-3; % integration time step
MAB_steps = 300;
% p.maxVal_d = 1; % Value here cancels out in gradient calc



%% --------Hyperparameters + simulating --------------------

p.a = 1.1;      % Levy tail exponent
p.gam = 1;      % strength of the Levy noise
p.beta = 1;     % momentum term: amount of acceleration
p.sigma2 = 0.5 * [1,1,1];
p.maxVal_d = 1;

p.temp = 3;     % softmax temperature
p.l = 0.99;       % recency bias
p.T = 1.8e2;      % simulation time: integer multiple of MAB_steps pls

tic
[X,t,history,history_rad] = fHMC_dynMABGaussian(p,payoffs,MAB_steps);
toc

optimal = sum(max(payoffs,[],2));
regret = 1 - (sum(history(2,:))/optimal);
[cnt_unique, uniq] = hist(history(1,:),unique(history(1,:)));
disp('Proportion of samples + overall regret')
disp([cnt_unique/sum(cnt_unique),regret])

%% Plotting spatial walk
figure
plot(X(1,:),X(2,:),'.','markerSize',1)
% hist3(X')

axis square
%% Plotting
figure
subplot(3,1,1)
for plt = 1:length(p.rewardMu)
    hold on
    plot(payoffs(:,plt))
end
xlabel('Step')
ylabel('Mean payoff')
title('Mean payoff walk')
legend('Option 1','Option 2','Option 3');%,'Option 4');

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
legend('Well 1', 'Well 2', 'Well 3');%, 'Well 4');
xlim([0,300])
title('Depth history')

%% Looping stuff
res_list = [];

optimal = sum(max(payoffs,[],2));
tic
parfor n = 1:100 %length(maxval_list)
    [X,t,history,history_rad] = fHMC_dynMABGaussian(p,payoffs,MAB_steps);
    
    regret = 1 - (sum(history(2,:))/optimal);
%     [cnt_unique, uniq] = hist(history(1,:),unique(history(1,:)));
%     disp(['Proportion of samples + overall regret: ',num2str(n)])
%     disp([cnt_unique/sum(cnt_unique),regret])
    res_list = [res_list, regret];
end
toc
%%
figure
histogram(res_list,30,'Normalization','probability');
xlabel('Regret')
ylabel('Probability')
disp(['Average regret: ', num2str(mean(res_list)),' Std: ', num2str(std(res_list))])
disp(['Median regret: ', num2str(median(res_list))])


