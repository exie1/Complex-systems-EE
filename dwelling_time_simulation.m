%% Just checking the truncated wells are functioning
% Run this code block for 1 simulation, and check the end sampled landscape

% Defining well parameters
p.location = [0,0];     p.radius = 3;   p.depth = 10;

% Defining walker parameters
p.a = 1.5;      p.gam = 2;      p.beta = 1;

% Defining imulation parameters
p.dt = 1e-3;    p.T = 1e2;      avg = 1;

tic     % Execute simulation for truncated quadratic wells
[X,t] = fHMC_quadratic(p,avg,p.radius,p.depth);
toc

% Compute the mean and std of dwelling time 
[mean_dwt,std_dwt,total_dwt] = dwellingTime(X,p);
disp('Mean and std of dwelling time:')
disp([mean(mean_dwt),mean(std_dwt)])

% Plot sampled distribution
plotSampling(X,p)

% Plot histogram of dwelling times
in_times = dwellingDist(X,p);
figure
histogram(in_times,100)
xlabel('Dwelling time (s)')


%% Loopage: compute results array

% Defining well parameters
p.location = [0,0];    p.radius = 1;    p.depth = 10; 

% Defining walker parameters
p.a = 1.5;      p.gam = 2;      p.beta = 1;

% Defining simulation parameters 
p.dt = 1e-3;    p.T = 1e2;

% Initializing recording arrays
numStep = 4;
mean_dwt_array = zeros(1,numStep);
std_dwt_array = zeros(1,numStep);
error_mean = zeros(1,numStep);
error_std = zeros(1,numStep);

% radius_array = linspace(1,pi,numStep);  % Loop over radius or depth
depth_array = linspace(1,20,numStep);

parfor i = 1:numStep
    [X,t] = fHMC_quadratic(p,20,p.radius,depth_array(i));
    
    [mean_dwt,std_dwt,total_dwt] = dwellingTime(X,p);
    mean_dwt_array(i) = mean(mean_dwt);
    std_dwt_array(i) = mean(std_dwt);
    
    error_mean(i) = std(mean_dwt);
    error_std(i) = std(std_dwt);
    disp(i)
end

plotDwelling(depth_array,mean_dwt_array,std_dwt_array,error_mean,error_std)



%% Functions

function [X,t] = fHMC_quadratic(p,avg,radius,depth)

    p.radius = radius;
    p.depth = depth;
    T = p.T;
    a = p.a;
    dt = p.dt;%1e-3; %integration time step (s)
    dta = dt.^(1/a); %fractional integration step
    n = floor(T/dt); %number of samples
    t = (0:n-1)*dt;

    x = zeros(2,avg)+[0;0]; %initial condition for each parallel sim
    v = zeros(2,avg)+[1;1];
    ca = gamma(a-1)/(gamma(a/2).^2);

    X = zeros(2,n,avg);
    for i = 1:n        
        f = makePotential(x,p);  % x ca for fractional derivative

        dL = stblrnd(a,0,p.gam,0,[2,avg]); 
        r = sqrt(sum(dL.*dL,1)); %step length
        
        th = rand(1,avg)*2*pi;
        g = r.*[cos(th);sin(th)];

        % Stochastic fractional Hamiltonian Monte Carlo
        vnew = v + p.beta*f*dt;
        xnew = x + p.gam*f*dt + p.beta*v*dt + g*dta;
                
        x = xnew;
        v = vnew;
        x = wrapToPi(x); % apply periodic boundary to avoid run-away
        X(:,i,:) = x;
    end
end

function f = makePotential(x,p)
    % Find the log gradient of a quadratic landscape
    % Just make the gradient 0 past a certain radius.
    f = 0;
    for j = 1:length(p.depth)
        distx = x(1,:) - p.location(j,1);
        disty = x(2,:) - p.location(j,2);

        z = p.depth(j) * ((distx.^2 + disty.^2) / p.radius(j) - 1);

        % Gradient function
        grad = -2 * p.depth(j) .* [distx; disty] / p.radius(j);
%         f = f - grad .* (z <= 0) ./z;   % Log gradient for pdf
        f =  f + grad .* (z<=0);    % Normal gradient for force
    end
      
end

function [mean_duration,std_duration,total_duration] = dwellingTime(X,p)
    % Find mean duration the sampling point stays in a well
    avg = size(X,3);
    mean_duration = zeros(length(p.depth),avg);
    std_duration = zeros(length(p.depth),avg);
    total_duration = zeros(length(p.depth),avg);
    for j = 1:avg
        for i = 1:size(p.location,1)
            loc = p.location(i,:);
            dist_stim = sqrt((X(1,:,j)-loc(1)).^2 + (X(2,:,j)-loc(2)).^2);
            
            in_stim = [0,(dist_stim < p.radius(i)),0];
            exit_indices = find([false,in_stim]~=[in_stim,false]);
            
            in_stim_times = (exit_indices(2:2:end) - exit_indices(1:2:end-1))*p.dt;
            filt_stim_times = in_stim_times(in_stim_times>0);
            mean_duration(i,j) = mean(filt_stim_times);
            std_duration(i,j) = std(filt_stim_times);
            total_duration(i,j) = sum(filt_stim_times);
        end
    end
end

function in_stim_times = dwellingDist(X,p)
    for i = 1:size(p.location,1)
        loc = p.location(i,:);
        dist_stim = sqrt((X(1,:,1)-loc(1)).^2 + (X(2,:,1)-loc(2)).^2);

        in_stim = [0,(dist_stim < p.radius(i)),0];
        exit_indices = find([false,in_stim]~=[in_stim,false]);

        in_stim_times = (exit_indices(2:2:end) - exit_indices(1:2:end-1))*p.dt;
    end
end

%% Plotting functions

function plotSampling(X,p)
    figure
    subplot(1,3,1)
    H2 = histogram2(X(1,:,1),X(2,:,1),100);

    subplot(1,3,2)
    hold on
    H1 = histogram(X(1,:,1),100,'normalization','pdf');
    
    xx  = H1.BinEdges;
    z = -p.depth * (xx.^2/p.radius - 1);
%     area = -2*p.depth * ( p.radius.^2/3 - p.radius );
    z = z .* (z >= 0);
%     plot(xx,z/10*0.65,'linewidth',1.5) 

    subplot(1,3,3)
    imagesc(H2.Values)
%     viscircles(p.location,p.radius);
end

function plotDwelling(d_array, m_array,s_array,e_m,e_s)
    figure
    subplot(1,2,1)
    hold on
    plot(d_array,m_array)
    errorbar(d_array,m_array,e_m)
    xlabel('depth')
    ylabel('mean dwelling time')
    title('mean dwt v depth')

    subplot(1,2,2)
    hold on
    plot(d_array,s_array)
    errorbar(d_array,s_array,e_s)
    xlabel('depth')
    ylabel('std dwelling time')
    title('std of dwt v depth')
end