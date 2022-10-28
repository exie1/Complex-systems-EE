%% Restless bandits problem

clear p
payoffs = (csvread('payoffs\payoffs_restless_3arms_distinct2.csv')'+300)/100;
payoff_time = size(payoffs,1);

figure
hold on 
plot(payoffs)
xlabel('Trial')
ylabel('Payoff')
legend('Option 1','Option 2','Option 3')

%% Simu

% Setting the (arbitrary?) location and well parameters
% p.location = [-1,1;1,-1]*pi/2;
% p.sigma2 = [1,1]*0.3;
% p.depth = payoffs(1,:);
clear p

p.location = pi/2*setPoints(3,pi/2);
p.sigma2 = [1,1,1]*0.3;
p.depth = payoffs(1,:);

Id = [1,0;0,1];

p.dt = 1e-3;
p.T = 1e2;

p.temp = 0.01;
p.sw = 5;
p.n = 1;


% a = 1.5;  gam = 1.5;  beta = 0.5
[X,t,history_FNS,spatial_FNS,depth_FNS] = fHMC_MAB_sw(p,payoffs,1.5,1.5,0.5);
figure
plot(cumsum(history_FNS(2,:)), 'DisplayName', 'FNS performance','LineWidth',1)
hold on

[X2,~,history_FNS2,spatial_FNS2,depth_FNS2] = fHMC_MAB_sw(p,payoffs,2,1.5,0.5);
plot(cumsum(history_FNS2(2,:)), 'DisplayName', 'FNS w/ Gaussian performance')
legend('Location','NorthWest')
xlabel('Trial')
ylabel('Cumulative reward')
%% Plotting cumulative reward
figure
hold on
plot(cumsum(history_FNS(2,:)), 'DisplayName', 'FNS performance','LineWidth',1)
plot(cumsum(history_FNS2(2,:)), 'DisplayName', 'FNS w/ Gaussian performance')
legend('Location','NorthWest')
xlabel('Trial')
ylabel('Cumulative reward')

%% Plotting choices

figure
subplot(3,1,1)
plot(payoffs)
legend('Option 1', 'Option 2', 'Option 3')

subplot(3,1,2)
plot(history_FNS(1,:),'DisplayName','FNS')
legend

subplot(3,1,3)
plot(history_FNS2(1,:),'DisplayName','FNS-G')
legend
   


%% Loopage for error bars and stuff
averages = 200;

history_FNS = zeros(averages,payoff_time);
history_FNSG = zeros(averages,payoff_time);

tic
parfor average = 1:averages
    [~,~,history,~,~] = fHMC_MAB_sw(p,payoffs,1.5,1,0.5);
    [~,~,historyG,~,~] = fHMC_MAB_sw(p,payoffs,2,1,0.5);

    history_FNS(average,:) = history(2,:);
    history_FNSG(average,:) = historyG(2,:);
end
toc

%%
figure
hold on

y1 = mean(cumsum(history_FNS,2),1)';
dy1 = std(cumsum(history_FNS,2),1)';

y2 = mean(cumsum(history_FNSG,2),1)';
dy2 = std(cumsum(history_FNSG,2),1)';

x = (1:size(history_FNS,2))';

plot(x,y1,'-','DisplayName','FNS performance','LineWidth',1.5)
plot(x,y2,'-.','DisplayName','FNS w/ Gaussian noise performance','LineWidth',1)


fill([x;flipud(x)],[y1-dy1 ; flipud(y1+dy1)],[0 0.4470 0.7410], ...
    'linestyle', 'none','FaceAlpha',0.4,'HandleVisibility','off')

fill([x;flipud(x)],[y2-dy2 ; flipud(y2+dy2)],[0.8500 0.3250 0.0980], ...
    'linestyle', 'none','FaceAlpha',0.2,'HandleVisibility','off')

legend('Location','NorthWest')

xlabel('Trial')
ylabel('Cumulative reward')
set(gca,'fontsize',14)
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





