%% Importing a [1,6] switching reward function

clear p
payoffs = (csvread('payoffs_randstep.csv')'+0.25)*0.8;
payoff_time = size(payoffs,1);

% Well parameters
p.location = pi/2*[-1,0;1,0];   p.sigma2 = [1,1]*0.3;   p.depth = payoffs(1,:);
% Simulation parameters
p.dt = 1e-3;    p.T = 1e2;      avg = 40;
% Sliding window parameters
p.temp = 0.01;  p.sw = 5;       p.n = 1;

tic
[~,t,~,spatial_FNS,~] = fHMC_MAB_sw(p,payoffs,1.5,1.5,1);
toc

history_FNS = zeros(avg,payoff_time);
history_dUCB = zeros(avg,payoff_time);
history_UCB = zeros(avg,payoff_time);
history_softmax= zeros(avg,payoff_time);

tic
parfor i = 1:avg
    [~,~,history0,~,~] = fHMC_MAB_sw(p,payoffs,1.5,1.5,0.5);
    
    [history1, ~] = dUCB(p,payoffs,0.96,0.3);
    [history2, ~] = UCB(p,payoffs,1);
    [history3, ~] = softmaxSim(p,payoffs,1);
    
    history_FNS(i,:) = history0(2,:);
    history_dUCB(i,:) = history1(2,:);
    history_UCB(i,:) = history2(2,:);
    history_softmax(i,:) = history3(2,:);
end
toc

histories = {history_FNS,history_dUCB,history_UCB,history_softmax};
labels = {'FNS','dUCB','UCB','Softmax'};
colours = {[0 0.4470 0.7410],[0.8500 0.3250 0.0980],...
        [0.9290 0.6940 0.1250],[0.4940 0.1840 0.5560]};
thick = {1.5,1,1,1};

clear history_FNS history_dUCB history_UCB history_softmax

%% Plotting: (a) payoffs, (b) spatial location (c) performance

subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.07], [0.15 0.01], [0.1 0.05]);
fnts = 16;

figure
subplot(2,2,1)
    hold on
    plot(payoffs,'LineWidth',1')
    legend('Option 1','Option 2')
    ylabel('Payoff')
    ylim([-0.3,1.3])
    set(gca,'fontsize', fnts,'xticklabel',[])

subplot(2,2,3)
    hold on
    plot(linspace(t(1),1000,payoff_time), squeeze(spatial_FNS(1,:)),'LineWidth',1)
    plot(linspace(t(1),1000,payoff_time),pi/2*((payoffs(:,2)/0.8 -0.25)*2-1),'LineWidth',2)

    legend('Sampled points','Ideal location')
    xlabel('Trial')
    ylabel('x coordinate')
    ylim([-3.9,3.9])
    set(gca,'fontsize', fnts)


subplot(2,2,[2,4])
    hold on
    for i = 1:4
        y1 = mean(cumsum(histories{i},2),1)';
        dy1 = std(cumsum(histories{i},2),1)';
        x = (1:size(histories{i},2))';

        plot(x,y1, 'color',colours{i}, 'DisplayName',labels{i},...
            'LineWidth',thick{i})
        fill([x;flipud(x)],[y1-dy1 ; flipud(y1+dy1)],colours{i}, ...
            'linestyle', 'none','FaceAlpha',0.2,'HandleVisibility','off')
    end
    legend('Location','NorthWest')
    xlabel('Trial')
    ylabel('Cumulative reward')
    ylim([0,inf])
    set(gca,'FontSize',fnts)
    


%% Extracting reward metrics
figure
hold on
for i = 1:4
    histogram(sum(histories{i},2),15,'DisplayName',labels{i})
    xlabel('Total reward')
    ylabel('Frequency')

    disp(labels{i})
    disp(mean(sum(histories{i},2)))
    disp(std(sum(histories{i},2)))
end
legend()
set(gca,'FontSize',fnts)


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
                                p.sigma2(chosen_option)*Id,1);
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
        peak = mvnpdf(p.location(i,:), p.location(i,:),p.sigma2(i)*[1,0;0,1]);
        reward = reward + payoff(i)*mvnpdf(coords, ...
                p.location(i,:),p.sigma2(i)*[1,0;0,1])/peak;
    end
end

function weights = softmax1(vec,temp)
    % Softmax function
    weights = exp(vec/temp)/sum(exp(vec/temp));
end
