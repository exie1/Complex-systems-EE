%% Importing an sampling a dynamic reward landscape
% The payoff increments every 500 or so simulation steps
clear p
payoffs = (csvread('payoffs\payoffs_switching_irregular.csv')')*5;
payoff_time = size(payoffs,1);


%%

% Setting the (arbitrary?) location and well parameters
p.location = [-1,1;1,-1];
p.sigma2 = [0.3,0.3];
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
%%  Plotting the switching behaviour
figure
subplot(2,1,1)
hold on
plot(depth_history(1,:))
plot(depth_history(2,:))

subplot(2,1,2)
hold on
plot(t,X(1,:))
plot(linspace(1,t(end),payoff_time),(payoffs(:,2)*2-1)./abs(payoffs(:,2)*2-1),'LineWidth',2)
xlabel('time')
ylabel('x position')

% plotMesh(p)

%% Compute optimal solution and plot
% [optimal_value,optimal_choice] = max(payoffs,[],2);
% 
% optimal_samples = mvnrnd(p.location(optimal_choice,:), ...
%                     p.sigma2(optimal_choice)*Id,length(t));
% optimal_reward = payoffFunction(optimal_samples,p);
% 
% regret = 1 - sum(reward)/sum(optimal_reward);
% 
figure
hold on
histogram2(X(1,:),X(2,:),50,'FaceAlpha',0.5);
histogram2(optimal_samples(:,1),optimal_samples(:,2),50,'FaceAlpha',0.1)


%% Evaluate MAB algorithm: first on total time, then reduce trials down
numWells = length(p.sigma2);
MAB_time = 500;
discount = 0.95;
balance = 0.3;

history = zeros(2,MAB_time);   % Row 1 = choice, row 2 = reward
discounted_history = zeros(1,MAB_time);
spatial_history = zeros(2,MAB_time);

for option = 1:numWells % Sample each option once
    sampled_point = mvnrnd(p.location(option,:),p.sigma2(option)*Id,1);
    sampled_reward = payoffStatic(sampled_point,p,payoffs(1,:));
    
    history(1,option) = option;
    history(2,option) = sampled_reward;
    discounted_history(option) = sampled_reward;
end

EV = zeros(1,numWells);
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
    [max_metric,chosen_option] = max(EV + balance*IB);
    sampled_point = mvnrnd(p.location(chosen_option,:), ...
                            p.sigma2(chosen_option)*Id,1);
    sampled_reward = payoffStatic(sampled_point,p,payoffs(trial,:));
    
    % ==== Update history and discounted history ====
    history(:,trial) = [chosen_option ; sampled_reward];
    discounted_history(trial) = sampled_reward;
    selected_history = (history(1,1:trial)==chosen_option);
    discounted_history(selected_history) = discounted_history(selected_history)*discount;
    
    spatial_history(:,trial) = sampled_point';
end

figure
subplot(2,1,1)
hold on
plot(payoffs(:,1))
plot(payoffs(:,2))
legend
subplot(2,1,2)
plot(history(1,:))

disp(mean(history(2,:)))

%% Testing
reward = payoffDynamic(X',p,payoffs);
figure
subplot(3,1,1)
plot(payoffs)

subplot(3,1,2)
% plot(t,X(1,:))
plot(reward(2,:))

subplot(3,1,3)
plot(spatial_history(1,:))


figure
hold on 
plot(cumsum(reward(1,:)),'DisplayName','FNS performance')
plot(cumsum(history(2,:)),'DisplayName','dUCB performance')

title('Cumulative reward gain for FNS and conventional MAB')
xlabel('Trial')
ylabel('Cumulative reward')
legend('Location','NorthWest')




%% Functions

function reward_array = payoffDynamic(coords,p,payoff)
    reward_array = zeros(2,size(payoff,1));
    step_ratio = size(coords,1) / size(payoff,1);
    
    for step = 0:size(payoff,1)-1
        reward = 0;
        window = step*step_ratio+1 : (step+1)*step_ratio;
        COM_window = [mean(coords(window,1)) , mean(coords(window,2))];
        for i = 1:length(p.sigma2)
            reward = reward + payoff(step+1,i) * ...
            mvnpdf(COM_window,p.location(i,:),p.sigma2(i)*[1,0;0,1]);
        end
        
        reward_array(1,step+1) = reward;
        reward_array(2,step+1) = COM_window(1);
        % Averaging across window is too much: more or less no change
        % "Throwing the baby out with the bath water"
        
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

function weights = softmax1(vec,temp)
    weights = exp(vec/temp) / sum( exp(vec/tem));
end