
clear p
payoffs = (csvread('payoffs\payoffs_novel.csv')');
switching = csvread('payoffs\switching_novel.csv');
payoff_time = size(payoffs,1);

%%

% Setting the (arbitrary?) location and well parameters
p.location = setPoints(3,pi/2);
p.sigma2 = [1,1,1]*0.3;
p.depth = payoffs(1,:);
Id = [1,0;0,1];

p.a = 1.5;
p.gam = 2;
p.beta = 1;

p.dt = 1e-3;
p.T = 1e2;

tic
[X,t,depth_history] = fHMC_optDynamic(p,1,payoffs);
toc

%%  Plotting the switching behaviour of FNS
figure
subplot(2,1,1)
hold on
plot(depth_history(1,:))
plot(depth_history(2,:))
plot(depth_history(3,:))

subplot(2,1,2)
hold on
plot(t,X(1,:))

%% Testing dUCB and UCB

res1 = dUCB(p,0.93,1,payoffs,switching);
history1 = res1{1};
spatial_history1 = res1{2};

res2 = UCB(p,5,payoffs,switching);
history2 = res2{1};
spatial_history2 = res2{2};

res3 = softmaxSim(p,0.5,payoffs,switching);
history3 = res3{1};
spatial_history3 = res3{2};

% plotChoices(history3,payoffs)
% disp(mean(history3(2,:)))

% === Comparing performance of FNS and dUCB + ... ===

reward = payoffDynamic(X',p,payoffs);
% plotChoices_spatial(reward,payoffs,spatial_history1,spatial_history2)

plotCumulative(reward,history1,history2,history3);



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

function reward_array = payoffDynamicMean(coords,p,payoff)
    % Collect reward at mean location
    reward_array = zeros(3,size(payoff,1));
    step_ratio = size(coords,2) / size(payoff,1);
    
    for step = 0:size(payoff,1)-1
        reward = 0;
        s_time = step*step_ratio+1 : (step+1)*step_ratio;
        s_point = [mean(coords(1,s_time)) , mean(coords(2,s_time))];
        
        for i = 1:length(p.sigma2)
            reward = reward + payoff(step+1,i) * ...
              mvnpdf(s_point,p.location(i,:),p.sigma2(i)*[1,0;0,1]);
        end
        
        reward_array(1,step+1) = reward;
        reward_array(2:3,step+1) = s_point';     
    end
end

function reward_array = payoffDynamicEnd(coords,p,payoff)
    % Collect reward at every specific point
    reward_array = zeros(3,size(payoff,1));
    step_ratio = size(coords,2) / size(payoff,1);
    
    for step = 0:size(payoff,1)-1
        reward = 0;
        s_time = (step+1/2)*step_ratio;
        s_point = coords(:,s_time);
        for i = 1:length(p.sigma2) 
            reward = reward + payoff(step+1,i) * ...
              mvnpdf(s_point',p.location(i,:),p.sigma2(i)*[1,0;0,1]);
        end
        
        reward_array(1,step+1) = reward;
        reward_array(2:3,step+1) = s_point';     
    end
end

function reward_array = payoffDynamicInd(coords,p,payoff)
    % Collect rewards independently from most sampled well
    reward_array = zeros(4,size(payoff,1));
    step_ratio = size(coords,2) / size(payoff,1);
    Id = [1,0;0,1];
    
    for step = 0:size(payoff,1)-1
        s_time = step*step_ratio+1 : (step+1)*step_ratio;
        [chosen_option,~] = proximityCheck(coords(:,s_time),p.location);
        
        sampled_point = mvnrnd(p.location(chosen_option,:), ...
                                p.sigma2(chosen_option)*Id,1);
        sampled_reward = payoffStatic(sampled_point,p,payoff(step+1,:));
        
        reward_array(1,step+1) = sampled_reward;
        reward_array(2:3,step+1) = sampled_point';     
        reward_array(4,step+1) = chosen_option;
    end
end

function reward = payoffStatic(coords,p,payoff)
    % Find payoff for each coordinate given the Gaussian parameters
    reward = 0;
    for i = 1:length(p.sigma2)
        reward = reward + payoff(i)*mvnpdf(coords, ...
                p.location(i,:),p.sigma2(i)*[1,0;0,1]);
    end
end

function res = dUCB(p,discount,balance,payoffs,switching)
    numWells = length(p.sigma2);
    MAB_time = 500;
    Id = [1,0;0,1];

    % Initialising history arrays
    history = zeros(2,MAB_time);    % Row 1 = choice, row 2 = reward
    discounted_history = zeros(1,MAB_time);
    spatial_history = zeros(2,MAB_time);

    for opt = 1:numWells % Sample each option once
        sampled_point = mvnrnd(p.location(opt,:),p.sigma2(opt)*Id,1);
        sampled_reward = payoffStatic(sampled_point,p,payoffs(1,:));

        history(1,opt) = opt;
        history(2,opt) = sampled_reward;
        discounted_history(opt) = sampled_reward;
    end

    EV = zeros(1,numWells);     % Initialising EV + IB of each option.
    IB = zeros(1,numWells);
    
    switch_c = [1,1,1];

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
            IB(opt) = sqrt(2*log(trial) ./ times_sampled);
        end
        
        % ==== Select the best option and sample from its distribution ====
        [~,chosen_option] = max(EV + balance*IB);
        sampled_point = mvnrnd(p.location(chosen_option,:), ...
                                p.sigma2(chosen_option)*Id,1);
        sampled_reward = payoffStatic(sampled_point,p,payoffs(trial,:));

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

function res = UCB(p,balance,payoffs,switching)
    numWells = length(p.sigma2);
    MAB_time = 500;
    Id = [1,0;0,1];

    % Initialising history arrays
    history = zeros(2,MAB_time);    % Row 1 = choice, row 2 = reward
    spatial_history = zeros(2,MAB_time);

    for option = 1:numWells % Sample each option once
        sampled_point = mvnrnd(p.location(option,:),p.sigma2(option)*Id,1);
        sampled_reward = payoffStatic(sampled_point,p,payoffs(1,:));

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
            IB(option) = sqrt(2*log(trial) ./ times_sampled);
        end
        
        % ==== Select the best option and sample from its distribution ====
        [~,chosen_option] = max(EV + balance*IB);
        sampled_point = mvnrnd(p.location(chosen_option,:), ...
                                p.sigma2(chosen_option)*Id,1);
        sampled_reward = payoffStatic(sampled_point,p,payoffs(trial,:));

        % ==== Update history ====
        history(:,trial) = [chosen_option ; sampled_reward];
        spatial_history(:,trial) = sampled_point';
    end
    res = {history,spatial_history};
    
end

function res = softmaxSim(p,temp,payoffs,switching)
    numWells = length(p.sigma2);
    MAB_time = 500;
    Id = [1,0;0,1];

    % Initialising history arrays
    history = zeros(2,MAB_time);    % Row 1 = choice, row 2 = reward
    spatial_history = zeros(2,MAB_time);

    for option = 1:numWells % Sample each option once
        sampled_point = mvnrnd(p.location(option,:),p.sigma2(option)*Id,1);
        sampled_reward = payoffStatic(sampled_point,p,payoffs(1,:));

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
        sampled_reward = payoffStatic(sampled_point,p,payoffs(trial,:));

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
function plotSwitching(X,t,payoffs)
    figure
    subplot(2,1,1)
    hold on
    for i = 1:2
        plot(payoffs(:,i))
    end

    subplot(2,1,2)
    hold on
    plot(t,X(1,:))
end

function plotChoices(history,payoffs)
    figure
    subplot(2,1,1)
    hold on
    plot(payoffs(:,1))
    plot(payoffs(:,2))
    plot(payoffs(:,3))
    legend

    subplot(2,1,2)
    hold on
    plot(history(1,:))
    legend
end

function plotChoices_spatial(reward,payoffs,spatial_history1,spatial_history2)
    figure
    subplot(3,1,1)
    plot(payoffs)
    legend
    ylabel('reward')

    subplot(3,1,2)
    % plot(t,X(1,:))
    plot(reward(2,:))
    ylabel('x pos')

    subplot(3,1,3)
    hold on
    plot(spatial_history1(1,:))
    plot(spatial_history2(1,:))
    xlabel('Trial')
    ylabel('x pos')
end
    
function plotCumulative(reward,history1,history2,history3)
    figure
    hold on 
    plot(cumsum(reward(1,:)),'DisplayName','FNS performance')
    plot(cumsum(history1(2,:)),'DisplayName','dUCB performance')
    plot(cumsum(history2(2,:)),'DisplayName','UCB performance')
    plot(cumsum(history3(2,:)),'DisplayName','softmax performance')
    
    title('Cumulative reward gain for 2 restless reward functions')
    xlabel('Trial')
    ylabel('Cumulative reward')
    legend('Location','NorthWest')
end