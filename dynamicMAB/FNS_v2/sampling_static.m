% Here we apply FNS on a static landscape, and test that the sampling
% matches up with the end-landscape. 


%% Applying FNS sampling on the reward landscape
clear p

p.location = [-1,1;0,0];
p.sigma2 = [0.3,0.3];
p.depth = [0,8];
Id = [1,0;0,1];

p.a = 1.5;
p.gam = 2;
p.beta = 1;

p.dt = 1e-3;
p.T = 1e2;

tic
[X,t] = fHMC_opt(p,1);
reward = payoffFunction(X',p);
toc

% plotMesh(p)

%% Compute optimal solution: only feed optimal well into FNS? 

optimal_choices = mvnrnd(p.location(2,:),p.sigma2(2)*Id,length(t));
optimal_reward = payoffFunction(optimal_choices,p);

regret = 1 - sum(reward)/sum(optimal_reward);
disp(regret)

figure
subplot(1,2,1)
hold on
bounds = linspace(-pi,pi,50);
histogram2(X(1,:),X(2,:),bounds,bounds,'FaceAlpha',0.5);
histogram2(optimal_choices(:,1),optimal_choices(:,2),bounds,bounds,'FaceAlpha',0.5)

subplot(1,2,2)
H3 = histogram(X(1,:),100,'normalization','pdf');
hold on
xx  = H3.BinEdges;
p2 = 1/sqrt(2*pi*p.sigma2(1))*exp(-0.5*xx.^2/p.sigma2(1));
plot(xx,p2,'LineWidth',2)


%% Evaluate MAB algorithm: first on total time, then reduce trials down
numWells = length(p.sigma2);
MAB_time = 100;
history = zeros(2,MAB_time);   % Row 1 = choice, row 2 = reward
for option = 1:numWells % Sample each option once
    sampled_point = mvnrnd(p.location(option,:),p.sigma2(option)*Id,1);
    sampled_reward = payoffFunction(sampled_point,p);
    
    history(1,option) = option;
    history(2,option) = sampled_reward;
end

EV = zeros(1,numWells);
IB = zeros(1,numWells);

for trial = numWells+1 : MAB_time
    for option = 1:numWells
        option_trials = (history(1,1:trial) == option);
        times_sampled = sum(option_trials);
        
        EV(option) = mean(option_trials.* history(2,option));
        IB(option) = sqrt(2*log(trial) ./ times_sampled);
    end
    
    [max_metric,chosen_option] = max(EV+IB);
    sampled_point = mvnrnd(p.location(chosen_option,:), ...
                            p.sigma2(chosen_option)*Id,1);
    sampled_reward = payoffFunction(sampled_point,p);
    
    history(:,trial) = [chosen_option ; sampled_reward];
end

optimal_choices2 = mvnrnd(p.location(2,:),p.sigma2(1)*Id,MAB_time);
optimal_reward2 = payoffFunction(optimal_choices2,p);
regret2 = 1 - sum(history(2,:))/sum(optimal_reward2);

figure
plot(history(1,:))

%% Functions
function reward = payoffFunction(coords,p)
    % Find payoff for each coordinate given the Gaussian parameters
    reward = 0;
    for i = 1:length(p.sigma2)
        reward = reward + p.depth(i)*mvnpdf(coords, ...
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