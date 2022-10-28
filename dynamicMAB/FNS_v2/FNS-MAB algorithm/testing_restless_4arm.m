clear p
payoffs = (csvread('payoffs\payoffs_restless_4arm.csv')'+300)/100;
payoff_time = size(payoffs,1);

% figure
% hold on 
% plot(payoffs)
% xlabel('Trial')
% ylabel('Payoff')
% legend('Option 1','Option 2','Option 3','Option 4')

p.location = pi/2*[-1,1;1,1;1,-1;-1,-1];    %3, pi
p.sigma2 = [1,1,1,1]*0.2;
p.depth = payoffs(1,:);

Id = [1,0;0,1];

p.dt = 1e-3;
p.T = 1e2;

p.temp = 0.08;
p.sw = 5;
p.n = 1;

tic
[X,t,history_FNS,spatial_FNS,depth_FNS] = fHMC_MAB_sw(p,payoffs,1.5,2,0);
toc


%%

[history_UCB, spatial_UCB] = UCB(p, payoffs, 1);

rewards_UCB = zeros([1,500]);
rewards_FNS = zeros([1,500]);
rewards_BEST = zeros([1,500]);


for i = 1:500
    option = 4;%history_UCB(1,i);
    sampled_point = mvnrnd(p.location(option,:),sqrt(p.sigma2(option))*Id,1);
    rewards_UCB(i) = generateReward(sampled_point,p,payoffs(i,:));
    
    optionFNS = 1;%history_FNS(1,i);
    sampled_rewardFNS = mvnrnd(p.location(optionFNS,:),sqrt(p.sigma2(optionFNS))*Id,1);
    rewards_FNS(i) = generateReward(sampled_rewardFNS,p,payoffs(i,:));
    
    optionBEST = ideal_choices(i);
    sampled_rewardBEST = mvnrnd(p.location(optionBEST,:),sqrt(p.sigma2(optionBEST))*Id,1);
    rewards_BEST(i) = generateReward(sampled_rewardBEST,p,payoffs(i,:));
end

figure 
hold on 
plot(cumsum(rewards_UCB))
plot(cumsum(rewards_FNS))
plot(cumsum(rewards_BEST))

legend('ucb','fns','best')


%%
figure
hold on
plot(payoffs(:,4))
plot(rewards_UCB)

%% Plotting pdf feature

numpts = 10;
xx = linspace(-pi,pi,numpts);
[XX,YY] = meshgrid(xx,xx);
co = [XX(:),YY(:)];
reward = zeros(numpts^2);

for i = 1:numpts^2
    reward(i) = mvnpdf([co(i,1),co(i,2)],p.location(4,:),sqrt(p.sigma2(4))*[1,0;0,1]);
end
figure
plot3(co(:,1),co(:,2),reward)
xlabel('x')
ylabel('y')

%% Plotting random samples from pdf
opt = 4;
samplepts = mvnrnd(p.location(opt,:),sqrt(p.sigma2(opt))*Id,10000);
rewards = mvnpdf(samplepts,p.location(opt,:),sqrt(p.sigma2(opt))*Id);
figure
hold on

plot3(samplepts(:,1),samplepts(:,2),rewards,'.')
surf(XX,YY,ZZ)

%%
function [history,spatial_history] = UCB(p,payoffs,balance)
    numWells = length(p.sigma2);
    MAB_time = size(payoffs,1);
    Id = [1,0;0,1];

    % Initialising history arrays
    history = zeros(2,MAB_time);    % Row 1 = choice, row 2 = reward
    spatial_history = zeros(2,MAB_time);

    for option = 1:numWells % Sample each option once
        sampled_point = mvnrnd(p.location(option,:),p.sigma2(option)*Id,1);
        sampled_reward = generateReward(sampled_point,p,payoffs(1,:));

        history(1,option) = option;
        history(2,option) = sampled_reward;
    end

    EV = zeros(1,numWells);     % Initialising EV + IB of each option.
    IB = zeros(1,numWells);

    for trial = numWells+1 : MAB_time
        % ==== Calculate the perceived value of each option ====
        for option = 1:numWells         
            option_trials = (history(1,1:trial) == option);  % Logical of each option
            times_sampled = sum(option_trials); % Times sampled of each option
            % Compute discounted expected value and information bonus
            EV(option) = mean(option_trials.* history(2,1:trial));
            IB(option) = sqrt(2*log(trial) ./ times_sampled);
        end
        
        % ==== Select the best option and sample from its distribution ====
        [~,chosen_option] = max(EV + balance*IB);
        sampled_point = mvnrnd(p.location(chosen_option,:),p.sigma2(chosen_option)*Id,1);
        sampled_reward = generateReward(sampled_point,p,payoffs(trial,:));

        % ==== Update history ====
        history(:,trial) = [chosen_option ; sampled_reward];
        spatial_history(:,trial) = sampled_point';
    end
end

function reward = generateReward(coords,p,payoff)
    % Find payoff for each coordinate given the Gaussian parameters
    % Should change to be an independent well vs the entire plane.
    reward = 0;
    for i = 1:length(p.sigma2)
        peak = mvnpdf(p.location(i,:), p.location(i,:),sqrt(p.sigma2(i))*[1,0;0,1]);
        reward = reward + payoff(i)*mvnpdf(coords,p.location(i,:),sqrt(p.sigma2(i))*[1,0;0,1])/peak;
    end
end