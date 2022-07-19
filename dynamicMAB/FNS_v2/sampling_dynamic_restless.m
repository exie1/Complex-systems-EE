%% Importing a restless reward function

clear p
payoffs = (csvread('payoffs\payoffs_2restlessA.csv')');
payoff_time = size(payoffs,1);

figure
hold on 
plot(payoffs(:,1))
plot(payoffs(:,2))

%%

% Setting the (arbitrary?) location and well parameters
p.location = [-1,1;1,-1];
p.sigma2 = [1,1]*0.3;
p.depth = payoffs(1,:);
Id = [1,0;0,1];

p.a = 1.2;
p.gam = 2;
p.beta = 1;

p.dt = 1e-3;
p.T = 5e1;

tic
[X,t] = fHMC_optDynamic(p,1,payoffs);
toc
%%
plotSwitching(X,t,payoffs)


%% Simulating standard MAB algorithms + generating FNS reward
[history1, spatial1] = dUCB(p,payoffs,0.95,1);
[history2, spatial2] = UCB(p,payoffs);
[history3, spatial3] = softmaxSim(p,payoffs,1);

reward = payoffDynamicInd(X(:,:,1),p,payoffs);

plotChoices(history1,spatial1,reward,payoffs)
plotCumulative(reward,history1(2,:),history2(2,:),history3(2,:));


figure
histogram2(reward(2,:),reward(3,:),15)

% plotChoices_spatial(X,t,reward,payoffs);
%%
time_window = 25:75;
[cnt_unique, uniq] = hist(reward(4,time_window),unique(reward(4,time_window)));
cnt_unique


%% Functions

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
    % Should change to be an independent well vs the entire plane.
    reward = 0;
    for i = 1:length(p.sigma2)
        reward = reward + payoff(i)*mvnpdf(coords, ...
                p.location(i,:),p.sigma2(i)*[1,0;0,1]);
    end
end

function mesh = generateMesh(points,bound)
    % Generate a meshgrid and convert into a coordinate list.
    [x,y] = meshgrid(linspace(-bound,bound,points), ...
            linspace(-bound,bound,points));
    mesh = [];
    for i = 1:size(x,1)^2
        mesh = [mesh, [x(i);y(i)]];
    end
end

function plotMesh(p)
    mesh = generateMesh(100,pi);
    y = payoffFunction(mesh,p);
    figure
    plot3(mesh(1,:),mesh(2,:),y)
end

function [history,spatial_history] = dUCB(p,payoffs,discount,balance)
    numWells = length(p.sigma2);
    MAB_time = 500;
    Id = [1,0;0,1];

    % Initialising history arrays
    history = zeros(2,MAB_time);    % Row 1 = choice, row 2 = reward
    discounted_history = zeros(1,MAB_time);
    spatial_history = zeros(2,MAB_time);

    for option = 1:numWells % Sample each option once
        sampled_point = mvnrnd(p.location(option,:),p.sigma2(option)*Id,1);
        sampled_reward = payoffStatic(sampled_point,p,payoffs(1,:));

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
        sampled_reward = payoffStatic(sampled_point,p,payoffs(trial,:));

        % ==== Update history and discounted history ====
        % For discounted: only for the chosen reward, so generate mask.
        history(:,trial) = [chosen_option ; sampled_reward];
        discounted_history(trial) = sampled_reward;
        selected_history = (history(1,1:trial)==chosen_option);
        discounted_history(selected_history) = discounted_history(selected_history)*discount;

        spatial_history(:,trial) = sampled_point';
    end
    
end

function [history,spatial_history] = UCB(p,payoffs)
    numWells = length(p.sigma2);
    MAB_time = 500;
    balance = 2;
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
        sampled_reward = payoffStatic(sampled_point,p,payoffs(trial,:));

        % ==== Update history ====
        history(:,trial) = [chosen_option ; sampled_reward];
        spatial_history(:,trial) = sampled_point';
    end
 
end

function [history,spatial_history] = softmaxSim(p,payoffs,temp)
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
        sampled_reward = payoffStatic(sampled_point,p,payoffs(trial,:));

        % ==== Update history ====
        history(:,trial) = [chosen_option ; sampled_reward];
        spatial_history(:,trial) = sampled_point';
    end
   
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

function plotChoices(history,spatial,reward,payoffs)
    figure
    subplot(3,1,1)
    hold on
    plot(payoffs(:,1))
    plot(payoffs(:,2))
    legend

    subplot(3,1,2)
    hold on
    plot(history(1,:))
    ylabel('choice')
    title('option selection')
    legend
    
    subplot(3,1,3)
    hold on
    plot(spatial(1,:),'DisplayName','dUCB performance')
    plot(reward(2,:),'DisplayName','FNS performance')
    ylabel('xpos')
    xlabel('trial')
    legend
    title('sampled reward points')
end

function plotChoices_spatial(X,t,reward,payoffs)
    figure
    subplot(3,1,1)
    plot(payoffs)
    legend
    xlabel('trial')
    ylabel('reward')
    title('Payoff function')

    subplot(3,1,2)
    hold on
    plot(reward(2,:))
%     plot(reward(4,:)*2-3)
    xlabel('trial')
    ylabel('x pos')
    title('FNS sampled reward points')
    
    subplot(3,1,3)
    plot(t,X(1,:))
    xlabel('time')
    ylabel('xpos')
    title('FNS sampled points')
end
    
function plotCumulative(reward,rewards1,rewards2,rewards3)
    figure
    hold on 
    plot(cumsum(reward(1,:)),'DisplayName','FNS performance')
    plot(cumsum(rewards1),'DisplayName','dUCB performance')
    plot(cumsum(rewards2),'DisplayName','UCB performance')
    plot(cumsum(rewards3),'DisplayName','softmax performance')

    title('Cumulative reward gain curves for 2 stepped reward functions')
    xlabel('Trial')
    ylabel('Cumulative reward')
    legend('Location','NorthWest')
end
