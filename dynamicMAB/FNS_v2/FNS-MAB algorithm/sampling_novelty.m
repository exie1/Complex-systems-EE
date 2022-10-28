clear p
payoffs = (csvread('payoffs\payoffs_novel.csv')');
switching = csvread('payoffs\switching_novel.csv');
payoff_time = size(payoffs,1);

figure
plot(payoffs)
xlabel('trial')
ylabel('payoff')
legend('1','2','3')
%% Running FNS simulation

% Setting the (arbitrary?) location and well parameters
p.location = setPoints(3,3*pi/2);
p.sigma2 = [1,1,1]*0.3;
p.depth = payoffs(1,:);
Id = [1,0;0,1];

p.a = 1.5;      p.gam = 2;  p.beta = 1;
p.temp = 0.05;  p.sw = 3;   p.n = 1;


p.dt = 1e-3; p.T = 1e2;

tic
[X,t,history_FNS,spatial_FNS,depth_FNS] = fHMC_sw_novelty(p,payoffs,switching);
toc

%% Comparing performance to conventional algorithms

history_dUCB = dUCB(p, payoffs, 0.97, 1, switching);
history_UCB = UCB(p, payoffs, 7, switching);
history_softmax = softmaxSim(p, payoffs, 1, switching);

plotChoices(payoffs,history_FNS,history_dUCB{1},history_UCB{1},history_softmax{1});
plotCumulative(history_FNS,history_dUCB{1},history_UCB{1},history_softmax{1})


%% Loopage



%% Functions

function points = setPoints(n,start)
    % Generate a regular set of wells on a circle around centre. 
    % Arrange on a unit circle by default, adjust spacing externally.
    
    w = 2*pi/n;                 % Angular distance between points
    points = zeros(n,2);        % Initialised location array
    
    for i = 0:n-1
        points(i+1,1) = cos(start - w*i);
        points(i+1,2) = sin(start - w*i);
    end
end

function reward = generateReward(coords,p,payoff)
    % Find payoff for each coordinate given the Gaussian parameters
    reward = 0;
    for i = 1:length(p.sigma2)
        reward = reward + payoff(i)*mvnpdf(coords, ...
                p.location(i,:),p.sigma2(i)*[1,0;0,1]);
    end
end

function res = dUCB(p,payoffs,discount,balance,switching)
    numWells = length(p.sigma2);
    MAB_time = 500;
    Id = [1,0;0,1];

    % Initialising history arrays
    history = zeros(2,MAB_time);    % Row 1 = choice, row 2 = reward
    discounted_history = zeros(1,MAB_time);
    spatial_history = zeros(2,MAB_time);

    for opt = 1:numWells % Sample each option once
        sampled_point = mvnrnd(p.location(opt,:),p.sigma2(opt)*Id,1);
        sampled_reward = generateReward(sampled_point,p,payoffs(1,:));

        history(1,opt) = opt;
        history(2,opt) = sampled_reward;
        discounted_history(opt) = sampled_reward;
    end

    EV = zeros(1,numWells);     % Initialising EV + IB of each option.
    IB = zeros(1,numWells);
    
    switch_c = [1,1,1];         % Compute history until last switch

    for trial = numWells+1 : MAB_time
        % ==== Start history anew if the option has switched ====
        match = find(trial == switching(1,:));
        if match
            switch_c(switching(2,match)+1) = trial;
        end
        
        
        % ==== Calculate the perceived value of each option ====
        for opt = 1:numWells         
            option_trials = (history(1,switch_c(opt):trial) == opt);  % Logical of each option
            times_sampled = sum(option_trials); % Times sampled of each option
            % Compute discounted expected value and information bonus
            EV(opt) = mean(option_trials.* discounted_history(switch_c(opt):trial));
            IB(opt) = sqrt(2*log(trial-switch_c(opt)) ./ times_sampled);
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
    res = {history,spatial_history};
    
end

function res = UCB(p,payoffs,balance,switching)
    numWells = length(p.sigma2);
    MAB_time = 500;
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
    switch_c = [1,1,1];
    
    for trial = numWells+1 : MAB_time
        % ==== Check for novel option, and reset history ====
        match = find(trial == switching(1,:));
        if match
            switch_c(switching(2,match)+1) = trial;
        end
        
        % ==== Calculate the perceived value of each option ====
        for option = 1:numWells         
            option_trials = (history(1,switch_c(option):trial) == option);  % Logical of each option
            times_sampled = sum(option_trials); % Times sampled of each option
            % Compute discounted expected value and information bonus
            EV(option) = mean(option_trials.* history(2,switch_c(option):trial));
            IB(option) = sqrt(2*log(trial-switch_c(option)) ./ times_sampled);
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
    res = {history,spatial_history};
    
end

function res = softmaxSim(p,payoffs, temp,switching)
    numWells = length(p.sigma2);
    MAB_time = 500;
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
    switch_c = [1,1,1];
    
    for trial = numWells+1 : MAB_time
        % ==== Check for novel option, and reset history ====
        match = find(trial == switching(1,:));
        if match
            switch_c(switching(2,match)+1) = trial;
        end
        
        % ==== Calculate the perceived value of each option ====
        for option = 1:numWells         
            option_trials = (history(1,switch_c(option):trial) == option);  % Logical of each option
            % Compute discounted expected value and information bonus
            EV(option) = mean(option_trials.* history(2,switch_c(option):trial));
        end
        
        % ==== Select the best option and sample from its distribution ====
        weights = softmax1(EV,temp);
        chosen_option = datasample(1:numWells,1,'Weights',weights);
        
        sampled_point = mvnrnd(p.location(chosen_option,:), ...
                                p.sigma2(chosen_option)*Id,1);
        sampled_reward = generateReward(sampled_point,p,payoffs(trial,:));

        % ==== Update history ====
        history(:,trial) = [chosen_option ; sampled_reward];
        spatial_history(:,trial) = sampled_point';
    end
    res = {history,spatial_history};
    
end

function weights = softmax1(vec,temp)
    weights = exp(vec/temp) / sum( exp(vec/temp));
end

%% Plotting functions

function plotChoices(payoffs,history_FNS,history_dUCB,history_UCB,history_softmax)
    figure
    subplot(5,1,1)
    plot(payoffs)

    subplot(5,1,2)
    plot(history_FNS(1,:),'DisplayName','FNS')
    legend
    
    subplot(5,1,3)
    plot(history_dUCB(1,:),'DisplayName','dUCB')
    legend
    
    subplot(5,1,4)
    plot(history_UCB(1,:),'DisplayName','UCB')
    legend
    
    subplot(5,1,5)
    plot(history_softmax(1,:),'DisplayName','softmax')
    legend
    
end

function plotCumulative(history_FNS,history_dUCB,history_UCB,history_softmax)
    figure
    hold on
    plot(cumsum(history_FNS(2,:)), 'DisplayName', 'FNS performance')
    plot(cumsum(history_dUCB(2,:)), 'DisplayName', 'dUCB performance')
    % plot(cumsum(history_swUCB(2,:)), 'DisplayName', 'swUCB performance')
    plot(cumsum(history_UCB(2,:)), 'DisplayName', 'UCB performance')
    plot(cumsum(history_softmax(2,:)), 'DisplayName', 'softmax performance')
    legend('Location','NorthWest')
    xlabel('Trial')
    ylabel('Cumulative reward')
end