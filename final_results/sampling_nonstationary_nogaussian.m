%% Restless bandits problem

clear p
payoffs = (csvread('payoffs\payoffs_restless_4arm.csv')'+300)/100;
payoff_time = size(payoffs,1);

figure
hold on 
plot(payoffs)
xlabel('Trial')
ylabel('Payoff')
legend('Option 1','Option 2','Option 3','Option 4')

%%

setPoints(2,3*pi/4)
pi/2 * setPoints(4,3*pi/4)

%% Simu

% Setting the (arbitrary?) location and well parameters
clear p

p.location = pi/2*setPoints(4,3*pi/4);%[-1,1;1,1;1,-1;-1,-1];    %3, pi
p.sigma2 = [1,1,1,1]*0.3;
p.depth = payoffs(1,:);

Id = [1,0;0,1];

p.dt = 1e-3;
p.T = 1e2;

p.temp = 0.05;
p.sw = 3;
p.n = 1;

tic
[X,t,history_FNS,spatial_FNS,depth_FNS] = fHMC_MAB_sw(p,payoffs,1.5,2,1);
toc

% plotSwitchingSpatial(X,t,payoffs,spatial_FNS)


%% Plotting the choices of FNS
figure
subplot(2,1,1)
hold on
plot(payoffs,'LineWidth',1)
xlabel('Trial')
ylabel('Payoff')
legend('Option 1', 'Option 2', 'Option 3','Option 4')
set(gca,'fontsize', 14)


subplot(2,1,2)
hold on
[~,best_option] = max(payoffs,[],2);
plot(1:500,best_option,'LineWidth',1.5,'color',[0.8500 0.3250 0.0980])
plot(1:500, history_FNS(1,:),'color',[0 0.4470 0.7410])

ylim([0.5,4.5])
xlabel('Trial')
ylabel('Choice taken')
set(gca,'fontsize', 14)

% ax = gca;
% ax.YAxis.TickValues = [1,2,3];
legend('Chosen options','Ideal option')

%% Checking performance against standard algorithms


[history_dUCB, spatial_dUCB] = dUCB(p, payoffs, 0.96, 0.3);   %0.96,3
% [history_swUCB, spatial_swUCB] = swUCB(p, payoffs, 5, 10);
[history_UCB, spatial_UCB] = UCB(p, payoffs, 1);
[history_softmax, spatial_softmax] = softmaxSim(p, payoffs, 1);


plotChoices(payoffs,history_FNS,history_dUCB,history_UCB,history_softmax);

plotCumulative(history_FNS,history_dUCB,history_UCB,history_softmax)


%% Loopage for error bars and stuff
averages = 100;

history_FNS = zeros(averages,payoff_time);
history_dUCB = zeros(averages,payoff_time);
history_UCB = zeros(averages,payoff_time);
history_softmax= zeros(averages,payoff_time);

tic
parfor average = 1:averages
    [~,~,history0,~,~] = fHMC_MAB_sw(p,payoffs,1.3,2,1);
    
    [history1, spatial1] = dUCB(p,payoffs,0.96,0.3);
    [history2, spatial2] = UCB(p,payoffs,1);
    [history3, spatial3] = softmaxSim(p,payoffs,1);
    
    history_FNS(average,:) = history0(2,:);
    history_dUCB(average,:) = history1(2,:);
    history_UCB(average,:) = history2(2,:);
    history_softmax(average,:) = history3(2,:);
end
toc

%%
plotCumulativeAvg(history_FNS,history_dUCB,history_UCB,history_softmax)
set(gca,'fontsize', 14)

%%
figure
subplot(1,2,1)
hold on
plot(mean(cumsum(history_FNS,2),1))
plot(mean(cumsum(history_dUCB,2),1))
legend('fns','ducb')

subplot(1,2,2)
hold on
plot(diff(mean(cumsum(history_FNS,2),1)))
plot(diff(mean(cumsum(history_dUCB,2),1)))
legend('fns','ducb')



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
    times_sampled = zeros(1,numWells);

    for trial = numWells+1 : MAB_time
        % ==== Calculate the perceived value of each option ====
        for option = 1:numWells         
            option_trials = (history(1,1:trial) == option);  % Logical of each option
%             times_sampled = sum(option_trials); % Times sampled of each option
            % Compute discounted expected value and information bonus
            EV(option) = mean(option_trials.* discounted_history(1:trial));
            IB(option) = sqrt(2*log(sum(times_sampled)) ./ times_sampled(option));
        end
        
        % ==== Select the best option and sample from its distribution ====
        [~,chosen_option] = max(EV + balance*IB);
        sampled_point = mvnrnd(p.location(chosen_option,:), ...
                                sqrt(p.sigma2(chosen_option))*Id,1);
        sampled_reward = generateReward(sampled_point,p,payoffs(trial,:));

        % ==== Update history and discounted history ====
        % For discounted: only for the chosen reward, so generate mask.
        history(:,trial) = [chosen_option ; sampled_reward];
        discounted_history(trial) = sampled_reward;
        selected_history = (history(1,1:trial)==chosen_option);
        discounted_history(selected_history) = discounted_history(selected_history)*discount;

        times_sampled(chosen_option) = times_sampled(chosen_option)*discount + 1;
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
        sampled_point = mvnrnd(p.location(chosen_option,:),sqrt(p.sigma2(chosen_option))*Id,1);
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
                                sqrt(p.sigma2(chosen_option))*Id,1);
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
        peak = mvnpdf(p.location(i,:), p.location(i,:),p.sigma2(i)*[1,0;0,1]);
        reward = reward + payoff(i)*mvnpdf(coords, ...
                p.location(i,:),sqrt(p.sigma2(i))*[1,0;0,1])/peak;
    end
end

function weights = softmax1(vec,temp)
    % Softmax function
    weights = exp(vec/temp)/sum(exp(vec/temp));
end

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

%% Plotting functions

function plotSwitchingChoices(X,t,payoffs,history_FNS,spatial_FNS,depth_FNS)
    payoff_time = size(payoffs,1); 

    figure
    subplot(3,1,1)
    hold on
    plot(payoffs)
    legend('Option 1', 'Option 2', 'Option 3')
    % plot(history_dUCB(2,:))

    subplot(3,1,2)
    hold on
    plot(t,X(1,:))
    plot(linspace(t(1),t(end),payoff_time), history_FNS(1,:))
%     plot(linspace(t(1),t(end),payoff_time), history_FNS(1,:)-2,'LineWidth',1)
    legend('Walker points','Sampled points')

    subplot(3,1,3)
    plot(depth_FNS')
end

function plotSwitchingSpatial(X,t,payoffs,spatial_FNS)
    payoff_time = size(payoffs,1); 

    figure
    subplot(2,1,1)
    hold on
    plot(payoffs,'LineWidth',1')
    legend('Option 1','Option 2')
    xlabel('Trial')
    ylabel('Payoff')
%     title('Payoffs over time')
    set(gca,'fontsize', 14)
    
%     payoffs = (payoffs-0.2)/0.8;
    
    subplot(2,1,2)
    hold on
%     plot(t,X(1,:),'LineWidth',1)
    
    [~,optimal_option] = max(payoffs,[],2);
    
    plot(linspace(t(1),500,payoff_time), spatial_FNS(1,:),'LineWidth',1)
    plot(linspace(t(1),500,payoff_time), ((optimal_option-1)*2-1)*pi/2,'LineWidth',2)

    legend('Sampled points','Ideal sampling')
    xlabel('Trial')
    ylabel('x coordinate')
    ylim([-pi,pi])
%     title('FNS sampler for switching payoff')
    set(gca,'fontsize', 14)
end

function plotChoices(payoffs,history_FNS,history_dUCB,history_UCB,history_softmax)
    figure
    subplot(5,1,1)
    plot(payoffs)
    legend('Option 1', 'Option 2', 'Option 3')
    
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
    plot(cumsum(history_FNS(2,:)), 'DisplayName', 'FNS performance','LineWidth',1)
    plot(cumsum(history_dUCB(2,:)), 'DisplayName', 'dUCB performance')
    plot(cumsum(history_UCB(2,:)), 'DisplayName', 'UCB performance')
    plot(cumsum(history_softmax(2,:)), 'DisplayName', 'softmax performance')
    legend('Location','NorthWest')
    xlabel('Trial')
    ylabel('Cumulative reward')
end

function plotCumulativeAvg(reward1,reward2,reward3,reward4)
    figure
    hold on

    y1 = mean(cumsum(reward1,2),1)';
    dy1 = std(cumsum(reward1,2),1)';
    
    y2 = mean(cumsum(reward2,2),1)';
    dy2 = std(cumsum(reward2,2),1)';
    
    y3 = mean(cumsum(reward3,2),1)';
    dy3 = std(cumsum(reward3,2),1)';
    
    y4 = mean(cumsum(reward4,2),1)';
    dy4 = std(cumsum(reward4,2),1)';
    
    x = (1:size(reward1,2))';
    
    plot(x,y1,'-','DisplayName','FNS performance','LineWidth',1.5)
    plot(x,y2,'-.','DisplayName','dUCB performance','LineWidth',1)
    plot(x,y3,'--','DisplayName','UCB performance','LineWidth',1)
    plot(x,y4,':','DisplayName','Softmax performance','LineWidth',1)
    
    fill([x;flipud(x)],[y1-dy1 ; flipud(y1+dy1)],[0 0.4470 0.7410], ...
        'linestyle', 'none','FaceAlpha',0.4,'HandleVisibility','off')
    
    fill([x;flipud(x)],[y2-dy2 ; flipud(y2+dy2)],[0.8500 0.3250 0.0980], ...
        'linestyle', 'none','FaceAlpha',0.2,'HandleVisibility','off')
    
    fill([x;flipud(x)],[y3-dy3 ; flipud(y3+dy3)],[0.9290 0.6940 0.1250], ...
        'linestyle', 'none','FaceAlpha',0.3,'HandleVisibility','off')
    
    fill([x;flipud(x)],[y4-dy4 ; flipud(y4+dy4)],[0.4940 0.1840 0.5560], ...
        'linestyle', 'none','FaceAlpha',0.1,'HandleVisibility','off')

    legend('Location','NorthWest')
    
    xlabel('Trial')
    ylabel('Cumulative reward')
end