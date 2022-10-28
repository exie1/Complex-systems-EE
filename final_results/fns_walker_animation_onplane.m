%% Simulating walker on 2D landscape
clear p
w.location = pi/2*[-1,1; 1,-1];     w.sigma2 = 0.3*[1,1];   w.depth = [1,2];
w.dt = 1e-3;    w.T = 3e1;      avg = 1;

tic
w.a = 1.2;      w.gam = 2;      w.beta = 3;
[X,t] = fHMC(w,avg);
toc

%%
figure
histogram(X(1,:))
%% Plot transparent 2D landscape
w.location = pi/2*[0,-1;0,1];%pi/2*setPoints(4,3*pi/4);     
w.sigma2 = 0.3*[1,1];   w.depth = [2,1];

figure('Position',[100,50,800,600])

arr = linspace(-pi,pi,50);
[xx,yy] = meshgrid(arr,arr);
zz = -landscape2D(arr,w);
s = surf(xx,yy,zz,'FaceAlpha',0.9,'Edgecolor','none');
hold on
% plot3(0,0,-0.6,'.','MarkerSize',25)

% view(5,25)
title('2D reward landscape')
xlim([-pi,pi]); ylim([-pi,pi]);
xlabel('x')
ylabel('y')
zlabel('Payoff')
set(gca,'FontSize',13)

%% Plot transparent 2D landscape with potential surface under.
w.location = pi/2*setPoints(3,pi/2);     
w.sigma2 = 0.3*[1,1,1];   w.depth = [1,2,3];

figure('Position',[100,50,800,600])

arr = linspace(-pi,pi,50);
[xx,yy] = meshgrid(arr,arr);
zz = -landscape2D(arr,w);
s = surf(xx,yy,zz,'FaceAlpha',0.9,'Edgecolor','none');
hold on
w2.location = w.location(3,:);    w2.sigma2 = 0.3;    w2.depth = 1;
zz2 = landscape2D(arr,w2);
plot3(xx,yy,zz2-0.05,'-k')
% plot3(0,0,-0.6,'.','MarkerSize',25)

% view(5,25)
% title('2D reward landscape')
xlim([-pi,pi]); ylim([-pi,pi]);
xlabel('x')
ylabel('y')
zlabel('Payoff')
legend('Reward landscape','Sampling landscape: Potential')
set(gca,'FontSize',13)

%% Animate walker on 2D landscape
figure('Position',[100,50,800,600])
ALines = animatedline;


Y = getReward(X',w);
frmC = 1;
skip = 15;
for i = 1:skip:length(t)
    endi = i + skip - 1;
    surfc(xx,yy,zz,'FaceAlpha',0.4,'Edgecolor','none');
    hold on
    plot3(X(1,i),X(2,i),Y(i),'.r','MarkerSize',30)
    plot3(X(1,i),X(2,i),-0.6,'.r','MarkerSize',10)
    hold off
%     addpoints(ALines,X(1,i:endi),X(2,i:endi),-0.6*ones(endi,1));
    view(5,25)
    xlabel('x')
    ylabel('y')
    zlabel('pdf')
    title(['FNS random walker: t = ', num2str(i*w.dt)])
    
    M(frmC) = getframe(gcf);
    frmC = frmC + 1;
    drawnow
end



%% Running movie separately
figure('Position',[100,50,800,600])
axes("Position",[0 0 1 1])
movie(M,1,30)
%% Saving movie
tic
walker_video = VideoWriter('C:\Users\Evan Xie\Desktop\planeWalker_exag', 'MPEG-4');

open(walker_video);
for m = 1:length(M)
    %img = readFrame(mov(m).cdata);
    writeVideo(walker_video, M(m));
end
close(walker_video);    
toc

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

function zz = landscape2D(arr,p)
    % Find payoff for each coordinate given the Gaussian parameters
    % Should change to be an independent well vs the entire plane.
    % -- Reward landscape rn is the pdf for each well * payoff
    zz = zeros(length(arr),length(arr));
    
    
    for x = 1:length(arr)
    for y = 1:length(arr)
        zz(x,y) = 0;
        co = [arr(x),arr(y)];
        for i = 1:length(p.sigma2)
            peak = mvnpdf(p.location(i,:), p.location(i,:),p.sigma2(i)*[1,0;0,1]);
            well = mvnpdf(co, p.location(i,:),p.sigma2(i)*[1,0;0,1]);
            zz(x,y) = zz(x,y) - p.depth(i)*well/peak;
        end
    end
    end
end

function r = getReward(co,p)
    r = 0;
    for i = 1:length(p.sigma2)
        r = r - mvnpdf(co, p.location(i,:),p.sigma2(i)*[1,0;0,1]);
    end
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

function reward = pdf_1D(x,p)
    reward = 0;
    for i = 1:length(p.sigma2)
        reward = reward - normpdf(x, p.location(i,1),p.sigma2(i));
    end
end


