%% Problem setup

clear p

% p.location = [-1,1;1,1;-1,-1;1,-1]*pi/2;
D = 1;
% p.location = D*[-1,1;1,1;-1,-1;1,-1];
% p.depth = [1,2,4,8];
% p.sigma2 = [1,1,1,1]*0.3;

p.location = pi/2 * [-1,0;1,0];
p.depth = [1,1];
p.sigma2 = 0.3*[1,1];

p.dt = 1e-3; % integration time step

p.a = 1.5;      % Levy tail exponent
p.gam = 2;      % noise strength
p.beta = 18;     % momentum term: amount of acceleration

p.T = 2e2;      % simulation time: integer multiple of MAB_steps pls


tic
[X,t] = fHMC_opt(p,1);
toc
%%
plotFFT(X,p,t)
% plotCoord(X,t)

% plotWalk(X,p)
% plotProp(X,p)

%%
seed = 12;       % Good seeds: 5, 6, 12
plotLevyWalk(200,p,seed)
%%
plotJumpDist(X)

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
    
    figure
    subplot(2,5,1:4)        % switching plot
    plot(t,X(1,:))
    xlabel('t')
    ylabel('x')

    
    subplot(2,5,5)          % histogram plot
    hist1 = histogram(X(1,:),50,'Normalization','pdf');
    xx = hist1.BinEdges;    p2 = 0;
    for i = 1:length(p.sigma2)
        p2 = p2 + p.depth(i)/sqrt(2*pi*p.sigma2(i))*...
            exp(-0.5*(xx-p.location(i,1)).^2/p.sigma2(i));
    end
    hold on
    plot(xx,p2/sum(p.depth),'LineWidth',1.5)
    xlim([-4,4])
    set(gca,'view',[90,90],'xdir','reverse');    % rotating figure
    set(gca,'YTickLabel',[],'XTickLabel',[]);   % removing axis labels

    
    subplot(2,5,6:10)       % FFT plot
    
    Y = fft(X(1,:));
    L = length(t);
    P2 = abs(Y/L);
    P1 = movmean(P2(1:L/2+1),30);   % FFT: sliding window of 30
    fs = 1/p.dt;                    % Sampling frequency
    f = (0:L/2)*fs/L;               % Frequecy range

    logx = log10(f);
    logy = log10(P1);
    filter = (logx>-1 & logx<0.3 | logx>1.3 & logx<2);
    f2 = polyfit(logx(filter),logy(filter),1);
    linefit = 10.^(f2(1)*logx(f>0.1)+f2(2));
    
    loglog(f,P1)
    hold on
    plot(f(f>0.1),linefit,'LineWidth',1.5)
    ylim([9e-6,1]);
    xlabel('frequency (Hz)')

end

function plotLevyWalk(time,p,seed)
    rng(seed)
    history = zeros(2,time);
    for t = 2:time
        dL = stblrnd(p.a,0,0.1,0,[2,1]); 
        r = sqrt(sum(dL.^2)); %step length
        
        th = rand*2*pi;
        g = r.*[cos(th);sin(th)];
        
        history(:,t) = history(:,t-1) + g;
    end
    
    figure
    plot(history(1,:),history(2,:),'.k');%,'MarkerEdgeColor','b',...)
    hold on
    x = history(1,:); y = history(2,:); z = 1:time;
    patch([x nan],[y nan],[z nan],[z nan], 'edgecolor', 'interp','LineWidth',0.8); 
    c = colorbar;colormap(jet);
    
    c.Label.String = 'step';
    xlabel('x')
    ylabel('y')
    axis('square')
    

end

function plotJumpDist(history)
    % Plot loglog distribution of jump size histogram
    jumps = diff(history,1,2);
    jumpsizes = sqrt(jumps(1,:).^2 + jumps(2,:).^2);
    [yy,xx] = histcounts(jumpsizes,100);
    
    xx = xx(yy>0); yy = yy(yy>0);
    
    fitlims = (xx>0.1 & xx<3);
    params = polyfit(log10(xx(fitlims)),log10(yy(fitlims)),1);
    linefit = 10.^(params(1)*log10(xx)+params(2));
    
    figure
    loglog(xx,yy)
    hold on
	loglog(xx(xx>0.1 & xx<3),linefit(xx>0.1 & xx<3),'LineWidth',1)
    xlim([min(xx),max(xx)])
    xlabel('Jump size')
    ylabel('Count')
end