% Generate movie of walker dynamics for 1D and 2D
% With and without momentum + test with Gaussian noise
% IDEA: either 1D or 2D cases, make a 2x2 plot for each of the above.

%% 2D walker dynamics 

% Potential landscape parameters
p.location = pi/2*[-1,1; 1,-1];     p.sigma2 = 0.3*[1,1];   p.depth = [2,1];

% Defining simulation parameters
p.dt = 1e-3;    p.T = 1.3e1;      avg = 1;

% 1e-3, 3e1

tic
% Walker parameters: normal
p.a = 1.3;      p.gam = 2;      p.beta = 1;
[X1,t] = fHMC(p,avg);
toc


% % Walker parameters: w/o mom
% p.a = 1.5;      p.gam = 1.5;      p.beta = 0;
% [X2,~] = fHMC(p,avg);
% 
% % Walker parameters: w/o levy
% p.a = 2;      p.gam = 1.5;      p.beta = 1;
% [X3,~] = fHMC(p,avg);
% 
% % Walker parameters: w/o mom and levy
% p.a = 2;      p.gam = 1.5;      p.beta = 0;
% [X4,~] = fHMC(p,avg);
% toc


% X = {X1,X2,X3,X4};
labels = {'\alpha = 1.5, \beta = 1', ...
        '\alpha = 1.5, \beta = 0', ...
        '\alpha = 2, \beta = 1', ...
        '\alpha = 2, \beta = 0'};
%% Preliminary histogram plotting
% plotHistory(X,labels);
figure
hold on
histogram(X1(1,:),100,'normalization','pdf')
xx = linspace(-pi,pi,100);
p2 = 0.5*1/sqrt(2*pi*p.sigma2(1))*exp(-0.5*(xx+pi/2).^2/p.sigma2(1));
p2 = p2 + 0.5*1/sqrt(2*pi*p.sigma2(2))*exp(-0.5*(xx-pi/2).^2/p.sigma2(2));
plot(xx,p2,'linewidth',1.5) 

%% Plot random choices
testsamples = randi([0,2],1,16);
figure
plot(0:15,testsamples,'or','LineWidth',1)

xlabel('Trial')
ylabel('Choice')
set(gca, 'YTick', 0:2)
set(gca,'fontsize', 14)
grid on


%% Plot FNS history

figure
plot(X1(1,:),X1(2,:),'k')
xlim([-pi,pi])
ylim([-pi,pi])
xlabel('x')
ylabel('y')
set(gca,'FontSize',13)



%% Animating FNS walker + histogram 

figure('Position',[100,50,600,900]);

subplot(2,1,1)
title('Trajectory of FNS random walker')
axis([-pi,pi,-pi,pi])
xlabel('x')
ylabel('y')

hold on
ALines = animatedline;
AScat = scatter(nan,nan,'r','filled');

subplot(2,1,2)
axis([-pi,pi,0,1.2])
xx = linspace(-pi,pi,100);
p2 = 0.5*1/sqrt(2*pi*p.sigma2(1))*exp(-0.5*(xx+pi/2).^2/p.sigma2(1));
p2 = p2 + 0.5*1/sqrt(2*pi*p.sigma2(2))*exp(-0.5*(xx-pi/2).^2/p.sigma2(2));

skip = 30;
frmC = 1;
for i = 1:skip:length(t)
    endi = i + skip - 1;
    addpoints(ALines,X1(1,i:endi),X1(2,i:endi));
    set(AScat,'xdata',X1(1,endi),'ydata',X1(2,endi));
    
    subplot(2,1,2)
    H3 = histogram(X1(1,1:i),80,'normalization','pdf');
    hold on
    plot(xx,p2,'linewidth',1.5) 
    hold off
    axis([-pi,pi,0,1.2])
    title(['Histogram of x-position: t = ',num2str(i*p.dt)])
    xlabel('x')
    ylabel('Density')
    
    
    M(frmC) = getframe(gcf);
    frmC = frmC + 1;
    drawnow
end


%% Animating FNS walkers on 2x2 grid
clear M 

figure('Position',[250,50,950,750]);

ALines = {};
AScat = {};
for i = 1:4
    subplot(2,2,i)
    title(labels{i})
    axis([-pi,pi,-pi,pi])
    xlabel('x')
    ylabel('y')
    
    sgtitle('Trajectory of FNS walker w/ different params')
    hold on
%     plot(p.location(1,1),p.location(1,2),'r*')
%     plot(p.location(1,1),p.location(1,2),'r*')
    ALines{i} = animatedline;
    AScat{i} = scatter(nan,nan,'r','filled');
end
    
skip = 20;
frmC = 1;
for i = 1:skip:length(t)
    endi = i + skip - 1;
    for j = 1:4
        addpoints(ALines{j},X{j}(1,i:endi),X{j}(2,i:endi));
        set(AScat{j},'xdata',X{j}(1,endi),'ydata',X{j}(2,endi));
        subplot(2,2,j)
        title([labels{j},': t = ', num2str(i*p.dt)])
    end
    M(frmC) = getframe(gcf);
    frmC = frmC + 1;
    drawnow
end


%% Animating reward landscape
clear M p

payoffs = (csvread('payoffs_novel_2active.csv')');

p.location = pi/2*setPoints(4,pi/4);
p.sigma2 = [1,1,1,1]*0.3;
p.depth = payoffs(1,:);

Alines = {};
colors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250],[0.4940 0.1840 0.5560]};
figure('Position',[100,50,600,900]);
    subplot(2,1,1)
    hold on
    plot(payoffs,'-.')
    for line = 1:4
        Alines{line} = animatedline('color',colors{line},'LineWidth',1);    
    end
    axis([0,size(payoffs,1),0,12])
    title('Payoff over all trials')
    xlabel('trial'); ylabel('Payoff');
    legend('Option 1','Option 2', 'Option 3','Option 4')

    
    
mesh = generateMesh(100,pi);
frmC = 1;
for i = 1:size(payoffs,1)
    subplot(2,1,2)
    y = payoffStatic(mesh',p,payoffs(i,:));
    plot3(mesh(1,:),mesh(2,:),y)
    hold on 
    for pt = 1:4
        plot3(p.location(pt,1),p.location(pt,2),11,'.',...
            'MarkerSize',20,'LineWidth',2,'color',colors{pt},...
            'DisplayName',['Option ',num2str(pt)])
    end
    hold off
    
    xlabel('x')
    ylabel('y')
    zlabel('Payoff')
    title(['Reward landscape: trial ', num2str(i)]) 
    
    for line = 1:4
        addpoints(Alines{line},i,payoffs(i,line))
    end

    M(frmC) = getframe(gcf);
    frmC = frmC + 1;
    drawnow
end

%% Running movie separately
figure('Position',[100,50,600,900]);
axes("Position",[0 0 1 1])
movie(M,1,45)
%% Saving movie

walker_video = VideoWriter('C:\Users\Evan Xie\Desktop\novel_payoffs', 'MPEG-4');

open(walker_video);
for m = 1:length(M)
    %img = readFrame(mov(m).cdata);
    writeVideo(walker_video, M(m));
end
close(walker_video);    

clear M
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

function [X,t] = fHMC(p,avg)

    T = p.T;
    a = p.a;
    dt = p.dt;%1e-3; %integration time step (s)
    dta = dt.^(1/a); %fractional integration step
    n = floor(T/dt); %number of samples
    t = (0:n-1)*dt;

    x = zeros(2,avg)+[0;0]; %initial condition for each parallel sim
    v = zeros(2,avg)+[0;1];
    ca = gamma(a-1)/(gamma(a/2).^2);

    X = zeros(2,n,avg);
    for i = 1:n        
        f = getPotential(x,p);  % x ca for fractional derivative

        dL = stblrnd(a,0,p.gam,0,[2,avg]); 
        r = sqrt(sum(dL.*dL,1)); %step length
        
        th = rand(1,avg)*2*pi;
        g = r.*[cos(th);sin(th)];

        % Stochastic fractional Hamiltonian Monte Carlo
        vnew = v + p.beta*ca*f*dt;
        xnew = x + p.gam*ca*f*dt + p.beta*v*dt + g*dta;
                
        x = xnew;
        v = vnew;
        x = wrapToPi(x); % apply periodic boundary to avoid run-away
        X(:,i,:) = x;
    end
end

function f = getPotential(x,p)
    % TARGET PDF DERIVATIVE CALCULATION (convert to fractional externally)
    fx = 0;
    fy = 0;     
    fn = 0;
    for j = 1:size(p.location,1) % optimise: compute x,y stuff together
        distx = x(1)-p.location(j,1);
        disty = x(2)-p.location(j,2);
        stim = p.depth(j) * exp(-0.5*(distx.^2+disty.^2)/p.sigma2(j));

        fx = fx + stim.*(-distx/p.sigma2(j));
        fy = fy + stim.*(-disty/p.sigma2(j));
        fn = fn + stim;
    end
    f = [fx; fy]./fn;  % log derivative
end

%% Plottage

function plotHistory(X,labels)
    figure
    for i = 1:4
        subplot(2,2,i)
        plot(X{i}(1,:),X{i}(2,:),'.','markersize',1)
        title(labels{i})
        xlim([-pi,pi]); ylim([-pi,pi]);
    end
end

function reward = payoffStatic(coords,p,payoff)
    % Find payoff for each coordinate given the Gaussian parameters
    % Should change to be an independent well vs the entire plane.
    % -- Reward landscape rn is the pdf for each well * payoff
    reward = 0;
    for i = 1:length(p.sigma2)
        pdf = mvnpdf(coords, p.location(i,:),p.sigma2(i)*[1,0;0,1]);
        reward = reward + payoff(i)*pdf/max(pdf);
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

function plotMesh(p,payoff)
    mesh = generateMesh(100,pi);
    y = payoffStatic(mesh',p,payoff);
    figure
    plot3(mesh(1,:),mesh(2,:),y)
end