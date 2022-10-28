%% 
p.location = pi/2*[-1,0;1,0];  p.sigma2 = 0.3*[1,1];   p.depth = [2,1];
p.a = 1.3;      p.gam = 1.5;       p.beta = 1;
p.dt = 1e-3;    p.T = 1.3e1;

tic
[X,t] = fHMC_opt(p,1);
toc


%% Plot the reward landscape
len = 50;
arr = linspace(-pi,pi,len);
[xx,yy] = meshgrid(arr,arr);
coords = [xx(:),yy(:)];
zz = reshape(getLandscape(coords,p),len,len);

fnts = 16;

clear subplot
figure('Position',[100,50,800,600])
subplot(1,2,1)
    grid on
    s = surf(xx,yy,zz,'FaceAlpha',0.9,'Edgecolor','none');
    view(5,25)
    xlim([-pi,pi]); ylim([-pi,pi]);
    xlabel('x')
    ylabel('y')
    zlabel('Payoff')
    set(gca,'FontSize',fnts)

subplot(1,2,2)
    plot(X(1,:),X(2,:),'k')
    xlim([-pi,pi])
    ylim([-pi,pi])
    xlabel('x')
    ylabel('y')
    set(gca,'FontSize',fnts)



% title('2D reward landscape')
% histogram2(X(1,:),X(2,:),arr,arr,'FaceAlpha',0.9,'Normalization','pdf');





%% Functions
function zz = getLandscape(coords,p)
    id = [1,0;0,1];
    m = size(coords,1);
    zz = zeros(m,1);
    for i = 1:m
        for j = 1:length(p.sigma2)
            peak = mvnpdf(p.location(j,:), p.location(j,:),p.sigma2(j)*id);
            well = mvnpdf(coords(i,:), p.location(j,:),p.sigma2(j)*id);
            zz(i) = zz(i) + p.depth(j)*well/peak;
        end
    end
end
