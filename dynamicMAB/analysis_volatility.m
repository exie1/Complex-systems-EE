% Analysing how the ability to switch wells changes when separation is
% decreased or if variance is increased.
% Consider: short but multiple simulations, and find std of time spent

%% One trial implementation
clear p 

radius = 0.7;
p.location = radius*setPoints(2, pi);
p.depth = [1,1];
p.sigma2 = [1,1]*0.3;

p.a = 1.3;      % Levy tail exponent
p.gam = 1;      % noise strength
p.beta = 1;     % momentum term: amount of acceleration

p.dt = 1e-3; % integration time step
p.T = 0.3e2;      % simulation time: integer multiple of MAB_steps pls


tic
[X,t] = fHMC_opt(p,1,pi/2);
toc

% plotWalk(X,p);
% plotCoord(X,t);
test = calcProp(X,p.location,true);



%% Looping business: first test separability of 2 wells
% Separation will go from 1 well width (min sep) to pi/2 (max sep)
sigma2 = 0.3;
p.sigma2 = sigma2*[1,1];
radius_list = linspace(sigma2,pi/2,200);
results_list = zeros(1,length(radius_list));

tic
parfor i = 1:length(radius_list)
    radius = radius_list(i);
    location = radius*setPoints(2, pi);
    
    [X,t] = fHMC_opt(p,1000,radius);
    
    props = calcProp(X,location,false);
    abs_diffs = abs(diff(props));
    results_list(i) = std(abs_diffs);
    disp([i,results_list(i)])
end
toc

figure
plot(radius_list,results_list)
xlabel('Well separation')
ylabel('Volatility')

%% Looping business: then test variance of wells
% Separation will go from 1 well width (min sep) to pi/2 (max sep)
p.location = [-1,0;1,0];

variance_list = logspace(-1,0,100);
results_list = zeros(1,length(variance_list));

tic
parfor i = 1:length(variance_list)
    sigma2 = variance_list(i);
    
    % Setting parameter changes inside fHMC to avoid parfor errors
    [X,t] = fHMC_opt(p,1000,sigma2);
    
    props = calcProp(X,p.location,false);
    abs_diffs = abs(diff(props));
    results_list(i) = std(abs_diffs);
    disp([i,results_list(i)])
end
toc
%%
figure
plot(variance_list,results_list)
xlabel('Well variance')
ylabel('Volatility')


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

function plotWalk(X,p)
    figure
    hold on
    plot(X(1,:),X(2,:),'.','markerSize',0.01)
    viscircles(p.location, sqrt(p.sigma2));
    plot(X(1,1),X(2,1),'og','lineWidth',1)
    plot(X(1,end),X(2,end),'or','lineWidth',1)
    
    xlim([-pi,pi])
    ylim([-pi,pi])
    xlabel('x')
    ylabel('y')
    axis square
end

function plotCoord(X,t)
    figure
    subplot(2,1,1)
    plot(t,X(1,:))
    xlabel('t')
    ylabel('x')
    subplot(2,1,2)
    plot(t,X(2,:))
    xlabel('t')
    ylabel('y')
end

function props = calcProp(X,location,plotCheck)
    [~,closest] = proximityCheck(X,location);
    props = zeros(size(location,1),size(closest,3));
    for i = 1:size(closest,3)
        [cnt_unique, ~] = hist(closest(:,:,i),unique(closest(:,:,i)));
        props(1:length(cnt_unique),i) = cnt_unique/sum(cnt_unique);
    end
    
    if plotCheck
        figure
        pie(categorical(closest))
    end

end