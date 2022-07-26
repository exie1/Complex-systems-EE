%% Importing a [1,6] switching reward function

clear p
payoffs = (csvread('payoffs_randstep.csv')')*5 + 1;
% payoffs = (csvread('payoffs\payoffs_2restlessC.csv')');
payoff_time = size(payoffs,1);

figure
hold on 
plot(payoffs(:,1))
plot(payoffs(:,2))

%% Simu

% Setting the (arbitrary?) location and well parameters
p.location = [-1,1;1,-1]*pi/2;
p.sigma2 = [1,1]*0.3;
p.depth = payoffs(1,:);
Id = [1,0;0,1];

p.a = 1.5;
p.gam = 2;
p.beta = 1;

p.dt = 1e-3;
p.T = 1e2;

p.temp = 0.01;
% p.l = 0.85;
p.sw = 3;
p.n = 2;


tic
[X,t,history_FNS,spatial_FNS,depth_FNS] = fHMC_MAB_sw(p,payoffs);
toc

%% Just checking switching behaviour
figure
subplot(3,1,1)
hold on
plot(payoffs)
% plot(history_dUCB(2,:))

subplot(3,1,2)
hold on
plot(t,X(1,:))
plot(linspace(t(1),t(end),payoff_time), spatial_FNS(1,:))
plot(linspace(t(1),t(end),payoff_time), history_FNS(1,:)*2-3,'LineWidth',1)

subplot(3,1,3)
plot(depth_FNS')

%% 
figure
subplot(2,1,1)
plot(payoffs)

subplot(2,1,2)
hold on
% plot(history_FNS(1,:),'DisplayName','FNS')
plot(history_UCB(1,:),'DisplayName','swUCB')
legend

%% Checking performance against standard algorithms


[history_dUCB, spatial_dUCB] = dUCB(p, payoffs, 0.95, 0.3);
[history_swUCB, spatial_swUCB] = swUCB(p, payoffs, 15, 10);
[history_UCB, spatial_UCB] = UCB(p, payoffs, 2);
[history_softmax, spatial_softmax] = softmaxSim(p, payoffs, 0.5);


figure
hold on
plot(cumsum(history_FNS(2,:)), 'DisplayName', 'FNS performance')
plot(cumsum(history_dUCB(2,:)), 'DisplayName', 'dUCB performance')
plot(cumsum(history_swUCB(2,:)), 'DisplayName', 'swUCB performance')
plot(cumsum(history_UCB(2,:)), 'DisplayName', 'UCB performance')
plot(cumsum(history_softmax(2,:)), 'DisplayName', 'softmax performance')
legend('Location','NorthWest')


%% Functions

function [history, spatial_history] = dUCB(p,payoffs,discount,balance)
    numWells = length(p.sigma2);
    MAB_time = size(payoffs,1);
    Id = [1,0;0,1];

    % Initialising history arrays
    history = zeros(2,MAB_time);    % Row 1 = choice, row 2 = reward
    discounted_history = zeros(1,MAB_time);
    spatial_history = zeros(2,MAB_time);

    for option = 1:numWells % Sample each option once
        sampled_point = mvnrnd(p.location(option,:),p.sigma2(option)*Id,1);
        sampled_reward = generateReward(sampled_point,p,payoffs(1,:));

        history(1,option) = option;
        history(2,option) = sampled_reward;
        discounted_history(option) = sampled_reward;
    end

    EV = zeros(1,numWells);     % Initialising EV + IB of each option.
    IB = zeros(1,numWells);

    for trial = numWells+1 : MAB_time
        % ==== Calculate the perceived value of each option ====
        for option = 1:numWells         
            option_trials = (history(1,1:trial) == option);  % Logical of each option
            times_sampled = sum(option_trials); % Times sampled of each option
            % Compute discounted expected value and information bonus
            EV(option) = mean(option_trials.* discounted_history(1:trial));
            IB(option) = sqrt(2*log(trial) ./ times_sampled);
        end
        
        % ==== Select the best option and sample from its distribution ====
        [~,chosen_option] = max(EV + balance*IB);
        sampled_point = mvnrnd(p.location(chosen_option,:), ...
                                p.sigma2(chosen_option)*Id,1);
        sampled_reward = generateReward(sampled_point,p,payoffs(trial,:));

        % ==== Update history and discounted history ====
        % For discounted: only for the chosen reward, so generate mask.
        history(:,trial) = [chosen_option ; sampled_reward];
        discounted_history(trial) = sampled_reward;
        selected_history = (history(1,1:trial)==chosen_option);
        discounted_history(selected_history) = discounted_history(selected_history)*discount;

        spatial_history(:,trial) = sampled_point';
    end
    
end

function [history,spatial_history] = swUCB(p,payoffs,sw,balance)
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
    times_sampled = ones(1,numWells);

    for trial = numWells+1 : MAB_time
        % ==== Calculate the perceived value of each option ====
        for option = 1:numWells         
            rewards_option = history(2,1:trial) .* (history(1,1:trial) == option);
            rewards_sw = rewards_option(rewards_option~=0);
            % Compute discounted expected value and information bonus
            EV(option) = mean(rewards_sw(max(1,end-sw):end));
        end
        IB = sqrt(2*log(trial) ./ times_sampled);
                    
        % ==== Select the best option and sample from its distribution ====
        [~,chosen_option] = max(EV + balance*IB);
        sampled_point = mvnrnd(p.location(chosen_option,:), ...
                                p.sigma2(chosen_option)*Id,1);
        sampled_reward = generateReward(sampled_point,p,payoffs(trial,:));
        times_sampled(chosen_option) = times_sampled(chosen_option) + 1;
        % ==== Update history ====
        history(:,trial) = [chosen_option ; sampled_reward];
        spatial_history(:,trial) = sampled_point';
    end
end

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
        sampled_point = mvnrnd(p.location(chosen_option,:), ...
                                p.sigma2(chosen_option)*Id,1);
        sampled_reward = generateReward(sampled_point,p,payoffs(trial,:));

        % ==== Update history ====
        history(:,trial) = [chosen_option ; sampled_reward];
        spatial_history(:,trial) = sampled_point';
    end
end

function [history,spatial_history] = softmaxSim(p,payoffs,temp)
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

    for trial = numWells+1 : MAB_time
        % ==== Calculate the perceived value of each option ====
        for option = 1:numWells         
            option_trials = (history(1,1:trial) == option);  % Logical of each option
            % Compute discounted expected value and information bonus
            EV(option) = mean(option_trials.* history(2,1:trial));
        end
        
        weights = softmax1(EV,temp);
        chosen_option = datasample(1:numWells,1,'Weights',weights);
        
        sampled_point = mvnrnd(p.location(chosen_option,:), ...
                                p.sigma2(chosen_option)*Id,1);
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
        reward = reward + payoff(i)*mvnpdf(coords, ...
                p.location(i,:),p.sigma2(i)*[1,0;0,1]);
    end
end

function weights = softmax1(vec,temp)
    % Softmax function
    weights = exp(vec/temp)/sum(exp(vec/temp));
end