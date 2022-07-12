clear p

points = linspace(-3*pi/5,3*pi/5,3);
p.location = build_lattice(points);
p.depth = p.location(:,1)*0 + 10;       % Start with 10 units of reward
p.sigma2 = ((pi/4+p.location(:,1)*0)/2).^2;

p.a = 1.5;
p.gam = 2;
p.beta = 1;

p.dt = 1e-3;
p.T = 1e2;
ths = 2*sqrt(p.sigma2);
avg = 1;
numsteps = 1e3;
discount_rate = 0.99;

tic
[X,t,history] = fHMC_foraging(p,avg,numsteps);
toc

[cnt_unique, uniq] = hist(history(2,:),unique(history(2,:)));
discounted_reward = sum(history(2,:) .* (discount_rate.^(1:numsteps)));


%% Functions


function [X,t,history] = fHMC_foraging(p,avg,numSteps)

%     p.location = rad * [-1,0;1,0];      % For parfor sim
%     p.sigma2 = sig2*[1,1];

    starting_reward = p.depth(1);
    dt = p.dt;%1e-3; %integration time step (s)
    dta = dt.^(1/p.a); %fractional integration step
    n = floor(p.T/dt); %number of samples
    t = (0:n-1)*dt;
    window = p.T/p.dt/numSteps;
    
    x = zeros(2,avg)+[0;0]; %initial condition for each parallel sim
    v = zeros(2,avg)+[1;1];
    ca = gamma(p.a-1)/(gamma(p.a/2).^2);

    X = zeros(2,n,avg);
    history = zeros(2,numSteps); % Row 1 = chosen well, row 2 = reward
    
    counter = 1;
    for i = 1:window:n      % num steps separated into time windows   
        % Executing fHMC simulation steps for time window 
        
        for w = i:i+window-1
            
            f = makeGaussian(x,p);

            dL = stblrnd(p.a,0,p.gam,0,[2,avg]); 
            r = sqrt(sum(dL.*dL,1)); %step length

            th = rand(1,avg)*2*pi;
            g = r.*[cos(th);sin(th)];

            % Stochastic fractional Hamiltonian Monte Carlo
            vnew = v + p.beta*ca*f*dt;
            xnew = x + p.gam*ca*f*dt + p.beta*v*dt + g*dta;

            x = xnew;
            v = vnew;
            x = wrapToPi(x); % apply periodic boundary to avoid run-away
            X(:,w,:) = x;
        end
       
        % Collect well depth as reward, decrement depth after
        [chosen,~] = proximityCheck(X(:,i:w),p.location);
        history(1,counter) = chosen; 
        
        if counter > 1
            if history(1,counter-1) == chosen % collect reward + decrement
                history(2,counter) = p.depth(chosen);
                p.depth(chosen) = p.depth(chosen) - 1;
            else % reset old well + don't sample from new well?
                p.depth(history(1,counter-1)) = starting_reward;
            end
        end
            
        counter = counter + 1;
    end

end


function f = makeGaussian(x,p)
    fx = 0;
    fy = 0;
    fn = 0;
    for j = 1:size(p.location,1) % optimise: compute x,y stuff together
        distx = x(1,:)-p.location(j,1);
        disty = x(2,:)-p.location(j,2);
        stim = p.depth(j) * exp(-0.5*(distx.^2+disty.^2)/p.sigma2(j));

        fx = fx + stim.*(-distx/p.sigma2(j));
        fy = fy + stim.*(-disty/p.sigma2(j));
        fn = fn + stim;
    end
    f = [fx; fy]./fn;  % log derivative: add e-15 to avoid 0/0
end


