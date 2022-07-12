%% Problem setup
payoffs = csvread('payoffs\payoffs_vol.csv')';

clear p

% Square: 4, 3*pi/4
% Triangle: 3, pi/2
% Pair: 2, pi

radius = 1;
p.location = radius*setPoints(2, pi);
p.depth = [1,1];

p.rewardMu = payoffs(1,:);
p.rewardSig = zeros(1,2) + 2;

p.dt = 1e-3; % integration time step
p.MAB_steps = 200;


%% --------Hyperparameters + simulating --------------------

p.a = 1.2;      % Levy tail exponent
p.gam = 1;      % strength of the Levy noise
p.beta = 0.5;     % momentum term: amount of acceleration
p.sigma2 = 0.4 * [1,1];
p.maxVal_d = 0.4;

p.temp = 1;     % softmax temperature
p.l = 0.97;       % recency bias
p.T = 1.5e2;      % simulation time: integer multiple of MAB_steps pls
p.softMin = 0.02;   % minimum depth from softmax
p.n = 2;        % Amount of directed exploration
% Don't set n too high or it gets exponentialled -> infinity
      
% Volatility stuff
p.swindow = 5;      % Sliding window for STD
p.maxseparation = pi/2;
p.minseparation = p.sigma2(1);
% Trying out a linear scale first


tic
[X,t,history,history_rad,history_sep] = fHMC_dynMABGaussian(p,payoffs);
toc

optimal = sum(max(payoffs,[],2));
regret = 1 - (sum(history(2,:))/optimal);
[cnt_unique, uniq] = hist(history(1,:),unique(history(1,:)));
disp('Proportion of samples + overall regret')
disp([cnt_unique/sum(cnt_unique),regret])


plotMAB(history,history_rad,p,payoffs)
% plotSpat(X,t)
% plotLoc(p.location)


%% Looping stuff
res_list = [];

optimal = sum(max(payoffs,[],2));
tic
parfor n = 1:100 %length(maxval_list)
    [X,t,history,history_rad] = fHMC_dynMABGaussian(p,payoffs,MAB_steps);
    
    regret = 1 - (sum(history(2,:))/optimal);
%     [cnt_unique, uniq] = hist(history(1,:),unique(history(1,:)));
%     disp(['Proportion of samples + overall regret: ',num2str(n)])
%     disp([cnt_unique/sum(cnt_unique),regret])
    res_list = [res_list, regret];
end
toc
%%
figure
histogram(res_list,30,'Normalization','probability');
xlabel('Regret')
ylabel('Probability')
disp(['Average regret: ', num2str(mean(res_list)),' Std: ', num2str(std(res_list))])
disp(['Median regret: ', num2str(median(res_list))])

%%

points = setPoints(4,3*pi/4);
disp(points)

figure
hold on
viscircles([0,0],1);
plot(points(:,1),points(:,2),'o','markersize',10,'linewidth',2);
plot(0,0,'.');

axis square

%% Plotting functions

function plotMAB(history,history_rad,p,payoffs)
    figure
    subplot(3,1,1)
    for plt = 1:length(p.rewardMu)
        hold on
        plot(payoffs(:,plt))
    end
    xlabel('Step')
    ylabel('Mean payoff')
    title('Mean payoff walk')
    legend('Option 1','Option 2');%,'Option 3','Option 4');

    subplot(3,1,2)
    plot(history(1,:),'.-')
    xlabel('Step')
    ylabel('Chosen option')
    title('Choice history')
    xlim([0,p.MAB_steps])

    subplot(3,1,3)
    for plt = 1:length(p.rewardMu)
        hold on
        plot(history_rad(plt,:))
    end
    xlabel('MAB step')
    ylabel('Depth')
    legend('Well 1', 'Well 2');%, 'Well 3', 'Well 4');
    xlim([0,200])
    title('Depth history')
end

function plotSpat(X,t)
    figure
    subplot(1,2,1)
    plot(X(1,:),X(2,:),'.','markerSize',0.01)
    axis square
    subplot(1,2,2)
    plot(t,X(1,:),'lineWidth',0.1)
    % hist3(X')
end

function plotLoc(points)
    disp(points)
    
    figure
    hold on
    viscircles([0,0],1);
    plot(points(:,1),points(:,2),'o','markersize',10,'linewidth',2);
    plot(0,0,'.');
    axis square

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
