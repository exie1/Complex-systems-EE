function [X,t,history,history_rad] = fHMC_dynMAB(p,payoffs,MAB_steps)
    % Same fHMC algorithm, but optimised for multi-armed bandit
    % simulations. Change width for bottom-up attention, change depth for
    % top-down attention, change Levy noise for random exploration.

    m = 2;           % dimensions
    dt = p.dt;%1e-3; % integration time step (s)
    dta = dt.^(1/p.a); % fractional integration step
    n = floor(p.T/dt); % number of samples
    t = (0:n-1)*dt;  % time
    window = p.T/p.dt/MAB_steps;
    max_rad = pi;
    numWells = length(p.rewardMu);
    
    
    x = zeros(2,1)+[1e-10;0]; % initial condition for each parallel sim
    v = zeros(2,1)+[1;0];     % ^ but velocity

    ca = gamma(p.a-1)/(gamma(p.a/2).^2);    % fractional derivative approx

    % Initialising recording arrays
    expectation = zeros(1,numWells);
    history = zeros(2,round(n/window)+numWells); 
    history_rad = zeros(numWells,round(n/window));
    X = zeros(m,n);
    
    % Sampling each well and adjusting radius
    history(:,1:numWells) = [1:numWells; p.rewardSig.*randn(1,numWells)+p.rewardMu];
    weights = softmax1(history(2,1:numWells),p.temp);
    p.radius2 = (max_rad*weights).^2;
    disp(weights)
    
    counter = 1+numWells;
    for i = 1:window:n      % num steps separated into time windows   
        for w = i:i+window-1
            f = getPotential(x,p);

            dL = stblrnd(p.a,0,p.gam,0,[2,1]); 
            r = sqrt(sum(dL.*dL,1)); %step length
            th = rand()*2*pi;
            g = r.*[cos(th);sin(th)];

            % Stochastic fractional Hamiltonian Monte Carlo
            vnew = v + p.beta*ca*f*dt;
            xnew = x + p.gam*ca*f*dt + p.beta*v*dt + g*dta;
            x = xnew;
            v = vnew;

            x = wrapToPi(x); % apply periodic boundary to avoid run-away
            X(:,w) = x;    % record position
        end
        chosen = proximityCheck(X(:,i:w),p.location);
        history(1,counter) = chosen;
        history(2,counter) = p.rewardSig(chosen)*randn()+p.rewardMu(chosen);
        history_rad(:,counter) = p.radius2';
        
        % Updating well parameters according to sampled history
        for opt = 1:length(p.rewardMu)
            rewards = history(2,1:counter) .* (history(1,1:counter) == opt);
            expectation(opt) = mean(rewards(rewards~=0));
        end
        weights = softmax1(expectation,p.temp);
        p.radius2 = (max_rad*weights).^2;
        p.depth = p.radius2; %sqrt or no?
        p.rewardMu = payoffs(counter-4,:);
        counter = counter + 1;
    end
end

function weights = softmax1(vec,temp)
    % Softmax function
    weights = exp(vec/temp)/sum(exp(vec/temp));
end

function f = getPotential(x,p)
    % FRACTIONAL DERIVATIVE CALCULATION
    fx = 0;
    fy = 0;     
    for j = 1:size(p.location,1)
        pot = p.depth(j)*(((x(1) - p.location(j,1)).^2 + ...
            (x(2) - p.location(j,2)).^2)/p.radius2(j) - 1);
        gradx = (x(1)-p.location(j,1))*2*p.depth(j)/p.radius2(j);
        grady = (x(2)-p.location(j,2))*2*p.depth(j)/p.radius2(j);
        fx = fx + gradx*(pot<=0);
        fy = fy + grady*(pot<=0);
    end
    f = -[fx;fy];
end


