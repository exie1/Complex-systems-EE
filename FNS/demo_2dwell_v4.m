function [X,t,MET]=demo_2dwell_v4()

%>>>>> parameters >>>>>>>
a = 1.2; % Levy tail exponent
p.sigma2 = 0.32; % sigma^2 where sigma is the width of the well
p.beta = 1; % beta coefficient
p.gam = 3; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e2; % total simulation time
p.location = pi/2; %modal location 
p.skip = 100; %number of time steps to skip for the video
animate = false; %set true to show video
%<<<<<< parameters <<<<<<<<

tic
disp('Generating raw data...')
[X,t] = fHMC_2d(T,a,p);

toc
   

n = floor(T/p.dt); %number of samples
t = (0:n-1)*p.dt;

close all

if animate
    show_video(X,t,p);
end

% [cl2r,cr2l] = closest_stim(X,t,p);
% c_tot = sort([cl2r,cr2l]);
% MET = mean(diff(c_tot));

figure('color','w');
subplot(2,1,1)
hold on
plot(t,X(1,:))
xlabel('t')
ylabel('x')
% plot(cl2r,zeros(length(cl2r)),'.')
% plot(cr2l,zeros(length(cr2l)),'o')

subplot(2,1,2)
plot(t,X(2,:))
xlabel('t')
ylabel('y')

figure('color','w');
plot(X(1,:),X(2,:),'.','markersize',1)
hold on
plot([1 -1]*p.location, [0 0],'*r')
xlabel('x')
ylabel('y')
axis equal

figure('color','w');
H = histogram2(X(1,:),X(2,:),100);
imagesc(H.Values)

figure('color','w')
subplot(2,1,1)
H2 = histogram(X(2,:),100,'normalization','pdf');
% hold on
% xx  = H2.BinEdges;
% p1 = 1/sqrt(2*pi*p.sigma2)*exp(-0.5*xx.^2/p.sigma2);
% %p2 = exp(-0.5*(x(1)+p.location).^2/p.sigma2 - 0.5*x(2).^2/p.sigma2 );
% plot(xx,p1,'linewidth',1.5) 
subplot(2,1,2)

subplot(2,1,2)
H3 = histogram(X(1,:),100,'normalization','pdf');
% hold on
% xx  = H3.BinEdges;
% p2 = 0.5*1/sqrt(2*pi*p.sigma2)*exp(-0.5*(xx+p.location).^2/p.sigma2);
% p2 = p2 + 0.5*1/sqrt(2*pi*p.sigma2)*exp(-0.5*(xx-p.location).^2/p.sigma2);
% plot(xx,p2,'linewidth',1.5) 

end

function [X,t] = fHMC_2d(T,a,p)
%Levy Monte Carlo on a circle
%INPUT: T = time span (s), a = Levy characteristic exponent
% p = other parameters
%OUTPUT: X = samples, t = time
if isempty(p)    
    p.location = pi/2;
    p.sigma2 = 1;
    p.gam = 1;
    p.dt = 1e-3;
    p.beta = 1;
end

m = 2; %dimension

dt = p.dt;%1e-3; %integration time step (s)
dta = dt.^(1/a); %fractional integration step

n = floor(T/dt); %number of samples
t = (0:n-1)*dt;
x = [0;0];%p.x0; %initial condition
v = [1;0];
X = zeros(m,n);

t0 = tic;
tic

%ca = gamma(a+1)/(gamma(a/2+1).^2); %<-- incorrect one
ca = gamma(a-1)/(gamma(a/2).^2); 

for i = 1:n
    
    %drift term
    % non-fractional grad of 2d gaussian wells
    p1 = exp(-0.5*(x(1)-p.location).^2/p.sigma2 - 0.5*x(2).^2/p.sigma2 );
    p2 = exp(-0.5*(x(1)+p.location).^2/p.sigma2 - 0.5*x(2).^2/p.sigma2 );
    
    fx = p1*(-(x(1)-p.location)/p.sigma2 ) + p2*(-(x(1)+p.location)/p.sigma2 );
    fy = -x(2)/p.sigma2;
    f = [fx./(p1+p2); fy]; % normalisation - get rid of
    
    dL = stblrnd(a,0,p.gam,0,[2 1]);
    r = sqrt(sum(dL.*dL)); %step length
    
    %g = [g1 ; g2];
    th = rand*2*pi;
    g = r*[cos(th);sin(th)];
   
    % Stochastic fractional hamiltonian monte carlo (2018)
    
    vnew = v + p.beta*ca*f*dt;
    xnew = x + p.gam*ca*f*dt + p.beta*v*dt + g*dta;
    
    x = xnew;
    v = vnew;
    
    x = wrapToPi(x); % apply periodic boundary to avoid run-away
    
    %x = wrapToPi(xnew); %periodic boundary condition
    if toc -t0 > 120
        disp('Time out!')
        %return
    end
    
    %endW
    X(:,i) = x;
    
end
end

function show_video(X,t,p)
figure('color','w');
for i = 1:p.skip:length(t)
    
    plot([1 -1]*p.location, [0 0],'*k')
    hold on
    plot(X(1,i),X(2,i),'o')
    hold off
    
    xlabel('x')
    ylabel('y')
    title(['t = ' num2str(t(i))])
    axis([-pi pi -pi pi])
    %axis equal   
    pause(1/60)   
end
end

function [c_stim_filt_l2r,c_stim_filt_r2l] = closest_stim(X,t,p)
    x_loc = [p.location, -p.location];
    dist_left = sqrt((X(1,:)-x_loc(1)).^2 + X(2,:).^2);
    dist_right = sqrt((X(1,:)-x_loc(2)).^2 + X(2,:).^2);
    c_stim = (dist_right>dist_left);
    c_stim_filt_l2r = t(strfind(c_stim,[0 1]));
    c_stim_filt_r2l = t(strfind(c_stim,[1 0]));
end
