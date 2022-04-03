function [X,t,history_rewards,history_choices] = fHMC_MAB_avgd(T,a,p,window,avg)
    % Same fHMC algorithm, but optimised for multi-armed bandit
    % simulations. Change width for bottom-up attention, change depth for
    % top-down attention, change Levy noise for random exploration.

    m = 2;           % dimensions
    dt = p.dt;%1e-3; % integration time step (s)
    dta = dt.^(1/a); % fractional integration step
    n = floor(T/dt); % number of samples
    t = (0:n-1)*dt;  % time
%     max_rad = vecnorm(p.location(1,:)-p.location(2,:))/2;
    max_rad = pi;
    numWells = length(p.rewardMu);
    
    
    x = zeros(avg,2)+[1e-10,0]; % initial condition for each parallel sim
    v = zeros(avg,2)+[0,1];     % ^ but velocity

    ca = gamma(a-1)/(gamma(a/2).^2);    % fractional derivative approx

    % Initialising recording arrays
    expectation = zeros(avg,numWells);
    history_rewards = zeros(avg,round(n/window)+numWells);
    history_choices = zeros(avg,round(n/window)+numWells);
    X = zeros(m,n,avg);
    
    % Sampling each well and adjusting radius
    for trial = 1:avg
        history_choices(trial,1:numWells) = 1:numWells;
        history_rewards(trial,1:numWells) = p.rewardSig.*randn(1,numWells)+p.rewardMu;
    end
    weights = softmax1(history_rewards(:,1:numWells));
    p.radius2 = (max_rad*weights).^2;
%     p.depth = sqrt(p.radius2); % sqrt? + don't initialise?
    disp(weights)
    
    counter = 1+numWells;
    for i = 1:window:n      % num steps separated into time windows   
        for w = i:i+window-1
            f = getPotential(x,p);

            dL = stblrnd(a,0,p.gam,0,[avg,2]); 
            r = sqrt(sum(dL.*dL,2)); %step length
            th = rand(avg,1)*2*pi;
            g = r.*[cos(th),sin(th)];

            % Stochastic fractional Hamiltonian Monte Carlo
            vnew = v + p.beta*ca*f*dt;
            xnew = x + p.gam*ca*f*dt + p.beta*v*dt + g*dta;
            x = xnew;
            v = vnew;

            x = wrapToPi(x); % apply periodic boundary to avoid run-away
            X(:,w,:) = x';    % record position
        end
        chosen = proximityCheck(X(:,i:w,:),p.location)';
        history_choices(:,counter) = chosen;
        history_rewards(:,counter) = p.rewardSig(chosen)'.*randn(avg,1) + p.rewardMu(chosen)';
                
        % Updating well parameters according to sampled history
        for opt = 1:length(p.rewardMu)
            rewards = history_rewards(:,1:counter).*(history_choices(:,1:counter)==opt);
            expectation(:,opt) = mean(rewards(rewards~=0),1);
        end
        weights = softmax1(expectation);
        p.radius2 = (max_rad*weights).^2;
        p.depth = p.radius2;
        counter = counter + 1;
    end
end

function weights = softmax1(vec)
    % Rowwise softmax function
    weights = exp(vec)./sum(exp(vec),2);
end

function f = getPotential(x,p)
    % FRACTIONAL DERIVATIVE CALCULATION
    fx = 0;
    fy = 0;     
    for j = 1:size(p.location,1)
        pot = p.depth(:,j).*(((x(:,1) - p.location(j,1)).^2 + ...
            (x(:,1) - p.location(j,2)).^2)./p.radius2(:,j) - 1);
        gradx = (x(:,1)-p.location(j,1))*2.*p.depth(:,j)./p.radius2(:,j);
        grady = (x(:,2)-p.location(j,2))*2.*p.depth(:,j)./p.radius2(:,j);
        fx = fx + gradx.*(pot<=0);
        fy = fy + grady.*(pot<=0);
    end
    f = -[fx,fy];
end
