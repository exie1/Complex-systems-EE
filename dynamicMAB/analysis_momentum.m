%% Problem setup

clear p

% p.location = [-1,1;1,1;-1,-1;1,-1]*pi/2;
D = 1;
p.location = D*[-1,1;1,1;-1,-1;1,-1];
p.depth = [1,2,4,8];
p.sigma2 = [1,1,1,1]*0.3;

p.dt = 1e-3; % integration time step

p.a = 2;      % Levy tail exponent
p.gam = 0.3;      % noise strength
p.beta = 0;     % momentum term: amount of acceleration

p.T = 3e2;      % simulation time: integer multiple of MAB_steps pls


tic
[X,t] = fHMC_opt(p,1);
toc

% plotFFT(X,p,t)
plotCoord(X,t)

plotWalk(X,p)
plotProp(X,p)

%% Plotting functions 

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

function plotProp(X,p)
    [~,closest] = proximityCheck(X,p.location);
    figure
    pie(categorical(closest))
    
    [cnt_unique, uniq] = hist(closest,unique(closest));
    disp(cnt_unique/sum(cnt_unique))
end

function plotFFT(X,p,t)
% Plotting Fourier transform of x coordinates in loglog
    Y = fft(X(1,:));
    L = length(t);

    P2 = abs(Y/L);
    P1 = movmean(P2(1:L/2+1),30);
    fs = 1/p.dt;        % Sampling frequency
    f = (0:L/2)*fs/L;

    figure
    subplot(2,1,1)
    plot(t,X(1,:))
    xlabel('t')
    ylabel('x')

    subplot(2,1,2)
    loglog(f,P1)
    ylim([9e-6,1]);
    xlabel('frequency (Hz)')

end



