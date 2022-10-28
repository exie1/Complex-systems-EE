%% Importing a stepped reward function
% The payoff increments every 500 or so simulation steps
clear p
payoffs = (csvread('payoffs_randstep.csv')'+0.25)*0.8;

payoff_time = size(payoffs,1);

% Well parameters
p.location = [-1,1;1,-1];   p.sigma2 = [1,1]*0.3;   p.depth = payoffs(1,:);
% Walker parameters
p.gam = 1.5;  p.beta = 1; Id = [1,0;0,1];
% Simulation parameters
p.dt = 1e-3;    p.T = 1e2;      avg = 400;


tic
[X,t] = fHMC_optDynamic(p,avg,payoffs,1.5);
[reward,spatial_FNS] = payoffDynamicEnd(X,p,payoffs);

[X2,~] = fHMC_optDynamic(p,avg,payoffs,2);
[reward2,spatial_FNS2] = payoffDynamicEnd(X2,p,payoffs);
toc

%%  Plotting the switching behaviour

plotSwitching(X2(:,:,1),t,payoffs,spatial_FNS(:,1,:))
% plotSwitching(X2(:,:,1),t,payoffs,spatial_FNS2(:,1,:))
% plotMesh(p)

%% Looping stuff
averages = 100;

history_dUCB = zeros(averages,payoff_time);
history_UCB = zeros(averages,payoff_time);
history_softmax= zeros(averages,payoff_time);

tic
parfor average = 1:averages
    [history1, spatial1] = dUCB(p,payoffs,0.98,0.4);
    [history2, spatial2] = UCB(p,payoffs,1);
    [history3, spatial3] = softmaxSim(p,payoffs,0.5);
    
    history_dUCB(average,:) = history1(2,:);
    history_UCB(average,:) = history2(2,:);
    history_softmax(average,:) = history3(2,:);
end
toc
%% Just plotting cumulative reward with Gaussian reward
plotCumulative(reward,history_dUCB,history_UCB,history_softmax);
set(gca,'FontSize',13)
% plotChoices(X(:,:,1),t,history1,spatial1,spatial_FNS,payoffs);

%% Comparing to Gaussian
compareGaussian(reward,reward2,history_dUCB)
ylim([0,250])
set(gca,'FontSize',13)

%%
plotMesh(p)

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

function [reward_array, sampled_array] = payoffDynamicEnd(coords,p,payoff)
    % Collect reward at every specific point
    reward_array = zeros(size(coords,3), size(payoff,1));
    sampled_array = zeros(2, size(coords,3), size(payoff,1));
    step_ratio = size(coords,2) / size(payoff,1);
    
    for step = 0:size(payoff,1)-1
        reward = 0;
        s_time = floor((step+1/2)*step_ratio); % Sample the midpoint
        s_point = squeeze(coords(:,s_time,:));
        for i = 1:length(p.sigma2) 
            reward = reward + payoff(step+1,i) * ...
              mvnpdf(s_point',p.location(i,:),p.sigma2(i)*[1,0;0,1]);
        end
        
        reward_array(:,step+1) = reward;
        sampled_array(:,:,step+1) = s_point;     
    end
end

function [reward_array, sampled_array] = payoffDynamicInd(coords,p,payoff)
    % Collect rewards independently from most sampled well
    reward_array = zeros(size(coords,3), size(payoff,1));
    sampled_array = zeros(2, size(coords,3), size(payoff,1));
    step_ratio = size(coords,2) / size(payoff,1);
    Id = [1,0;0,1];
    
    for step = 0:size(payoff,1)-1
        s_time = step*step_ratio+1 : (step+1)*step_ratio;
        [chosen_option,~] = proximityCheck(coords(:,s_time),p.location);
        
        sampled_point = mvnrnd(p.location(chosen_option,:), ...
                                p.sigma2(chosen_option)*Id,1);
        sampled_reward = payoffStatic(sampled_point,p,payoff(step+1,:));
        
        reward_array(:,step+1) = sampled_reward;
        sampled_array(:,:,step+1) = sampled_point; 
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
    y = payoffStatic(mesh',p,[1,1]);
    figure
    plot3(mesh(1,:),mesh(2,:),y)
end

function [history,spatial_history] = dUCB(p,payoffs,discount,balance)
    numWells = length(p.sigma2);
    MAB_time = size(payoffs,1);
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
    times_sampled = zeros(1,numWells)+1;

    for trial = numWells+1 : MAB_time
        % ==== Calculate the perceived value of each option ====
        for option = 1:numWells         
            option_trials = (history(1,1:trial) == option);  % Logical of each option
%             times_sampled = sum(option_trials); % Times sampled of each option
            % Compute discounted expected value and information bonus
            EV(option) = mean(option_trials.* discounted_history(1:trial));
            IB(option) = sqrt(2*log(trial) ./ times_sampled(option));
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
    MAB_time = size(payoffs,1);
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

function plotSwitching(X,t,payoffs,spatial_FNS)
    payoff_time = size(payoffs,1); 

    figure
    subplot(2,1,1)
    hold on
    plot(payoffs,'LineWidth',1')
    legend('Option 1','Option 2')
    xlabel('Trial')
    ylabel('Payoff')
    ylim([-0.3,1.3])
%     title('Payoffs over time')
    set(gca,'fontsize', 14)
    
%     payoffs = (payoffs-0.2)/0.8;
    
    subplot(2,1,2)
    hold on
%     plot(t,X(1,:),'LineWidth',1)
    plot(linspace(t(1),1000,payoff_time), spatial_FNS(1,:),'LineWidth',1)
    plot(linspace(t(1),1000,payoff_time),payoffs(:,2)*2-1,'LineWidth',2)

    legend('Sampled points','Ideal sampling')
    xlabel('Trial')
    ylabel('x coordinate')
    ylim([-pi,pi])
%     title('FNS sampler for switching payoff')
    set(gca,'fontsize', 14)
end

function plotChoices(X,t,history,spatial,spatial_FNS,payoffs)
    figure
    subplot(4,1,1)
    hold on
    plot(payoffs(:,1))
    plot(payoffs(:,2))
    legend

    subplot(4,1,2)
    hold on
    plot(history(1,:))
    ylabel('choice')
    title('chosen options')
    legend
    
    subplot(4,1,3)
    hold on
    plot(spatial(1,:),'DisplayName','dUCB sampled')
    plot(squeeze(spatial_FNS(1,1,:)),'DisplayName','FNS sampled' )
    ylabel('xpos')
    xlabel('trial')
    title('reward sample points dUCB')
    legend
    
    subplot(4,1,4)
    hold on
    plot(t,X(1,:,1),'DisplayName','FNS simulated')
    plot(linspace(0,t(end),1000),squeeze(spatial_FNS(1,1,:)),...
        'DisplayName','FNS sampled')
    xlabel('time')
    ylabel('xpos')
    title('simulated sample points')
    legend
    
    
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
    
function plotCumulative(reward1,reward2,reward3,reward4)
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
    
    plot(x,y1,'DisplayName','FNS performance','LineWidth',1.5)
    plot(x,y2,'DisplayName','dUCB performance')
    plot(x,y3,'DisplayName','UCB performance')
    plot(x,y4,'DisplayName','Softmax performance')
    
    fill([x;flipud(x)],[y1-dy1 ; flipud(y1+dy1)],[0 0.4470 0.7410], ...
        'linestyle', 'none','FaceAlpha',0.2,'HandleVisibility','off')
    
    fill([x;flipud(x)],[y2-dy2 ; flipud(y2+dy2)],[0.8500 0.3250 0.0980], ...
        'linestyle', 'none','FaceAlpha',0.2,'HandleVisibility','off')
    
    fill([x;flipud(x)],[y3-dy3 ; flipud(y3+dy3)],[0.9290 0.6940 0.1250], ...
        'linestyle', 'none','FaceAlpha',0.2,'HandleVisibility','off')
    
    fill([x;flipud(x)],[y4-dy4 ; flipud(y4+dy4)],[0.4940 0.1840 0.5560], ...
        'linestyle', 'none','FaceAlpha',0.2,'HandleVisibility','off')

    legend('Location','NorthWest')
    
    xlabel('Trial')
    ylabel('Cumulative reward')
end

function compareGaussian(reward_FNS,reward_FNSG,reward_dUCB)
        figure
    hold on

    y1 = mean(cumsum(reward_FNS,2),1)';
    dy1 = std(cumsum(reward_FNS,2),1)';
    
    y2 = mean(cumsum(reward_FNSG,2),1)';
    dy2 = std(cumsum(reward_FNSG,2),1)';
    
    y3 = mean(cumsum(reward_dUCB,2),1)';
    dy3 = std(cumsum(reward_dUCB,2),1)';
    

    x = (1:size(reward_FNS,2))';
    
    plot(x,y1,'DisplayName','FNS performance','LineWidth',1.5)
    plot(x,y2,'DisplayName','FNS-G performance','LineWidth',1.5)
    plot(x,y3,'DisplayName','dUCB performance')

    
    fill([x;flipud(x)],[y1-dy1 ; flipud(y1+dy1)],[0 0.4470 0.7410], ...
        'linestyle', 'none','FaceAlpha',0.2,'HandleVisibility','off')
    
    fill([x;flipud(x)],[y2-dy2 ; flipud(y2+dy2)],[0.8500 0.3250 0.0980], ...
        'linestyle', 'none','FaceAlpha',0.2,'HandleVisibility','off')
    
    fill([x;flipud(x)],[y3-dy3 ; flipud(y3+dy3)],[0.9290 0.6940 0.1250], ...
        'linestyle', 'none','FaceAlpha',0.2,'HandleVisibility','off')

    legend('Location','NorthWest')
    
    xlabel('Trial')
    ylabel('Cumulative reward')
end
