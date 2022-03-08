a = 1.2; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e2; % simulation time
num_parallel = 2;

%--------Creating lattice for stimulus locations----------------
boundaries = linspace(-pi+0.5,pi-0.5,3);
locations = build_lattice(boundaries);
depths = [0.8445;0.1969;0.8172;0.7434;0.1558;0.2688;0.8570;0.3224;0.0587];
% depths = rand(size(locations,1),1);
width = 0.005;
volume = sum(pi*depths/width);
%-----------------------------------------------------------------

p.location = locations;   
p.sigma2 = width + zeros(size(locations,1),1);
p.depth = depths/volume*10^3;       % Total reward = 10^3

tic
[X,t] = fHMC_opt(T,a,p,num_parallel);
toc

rewards = gaussian_rewards(X,p);
plot3(X(1,:,1),X(2,:,1),rewards(:,1),'.') % use regret/optimal score

rewards_total = sum(rewards,1)/size(X,2);
disp([mean(rewards_total),std(rewards_total)])
% max_reward = max(p.depth); 
%change max_reward to be sampling from optimal mode, instead of max value?

%% Looping business (for Levy noise)
num_parallel = 30;
% noise_list = logspace(-2,2,80);
% tail_list = linspace(1.1,1.9,80);
tail_list = 1+logspace(log10(0.01),log10(0.99),80);
rewards_list = [];
rewards_std = [];
tic
for j = 1:length(tail_list)
    a = tail_list(j);
    [X,t] = fHMC_opt(T,a,p,num_parallel);
    rewards = sum(gaussian_rewards(X,p))/size(X,2);
    rewards_list(j) = mean(rewards);
    rewards_std(j) = std(rewards);
    disp([j, rewards_list(j),rewards_std(j),tail_list(j)])
end
toc
%%
figure
plot(tail_list,rewards_list)
xlabel("Levy tail index")
ylabel("Total reward attained")
title('Tail index v total reward')



%%

function max_reward = maxReward(T,a,p)
    p_max = p;
    p_max.gamma = 2.5;
    p_max.location = [0,0];
    p_max.sigma2 = p.sigma2(1);
    p_max.depth = max(p.depth);
    
    [X_max,~] = fHMC_opt(T,a,p_max,1);
    plot(X_max(1,:),X_max(2,:),'.')
    max_reward = mean(sum(gaussian_rewards(X_max,p_max),1)/size(X_max,2));
end

