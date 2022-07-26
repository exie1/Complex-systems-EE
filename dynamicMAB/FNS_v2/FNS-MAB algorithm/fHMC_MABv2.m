function [X,t,history,reward_points,d_history] = fHMC_MABv2(p,payoffs)
    % Coupling the FNSv2 scheme to proper non-stationary MAB problems. 
    % We use the dUCB + softmax scheme now to artifically change
    % the sampled proportions of FNS. 

    
% ============================================================
%             Initialization p1: params + recording 
% ============================================================

    dt = p.dt;           % integration time step (s)
    dta = dt.^(1/p.a);   % fractional integration step
    [MAB_time, numWells] = size(payoffs);
    window = p.T/p.dt/MAB_time;
    n = floor(p.T/dt) - numWells*window;   % number of samples
    t = (0:n-1)*dt;      % time
    
    Id = [1,0;0,1];
    
    x = zeros(2,1); % initial condition for each parallel sim
    v = zeros(2,1)+[1;1];     % ^ but velocity

    ca = gamma(p.a-1)/(gamma(p.a/2).^2);    % approx. fractional derivative

    % Initialising recording arrays
    EV = zeros(1,numWells);
    history = zeros(2,MAB_time); 
    history_discounted = zeros(1,MAB_time);
    reward_points = zeros(2,MAB_time);
    d_history = zeros(numWells,MAB_time);
    X = zeros(2,n);
    
% ============================================================
%               Initialization p2: MAB start
% ============================================================
    
    % --- Artifically sample once from each option ---
    for option = 1:numWells     
        sampled_point = mvnrnd(p.location(option,:),p.sigma2(option)*Id,1);
        sampled_reward = generateReward(sampled_point,p,payoffs(option,:));

        history(:,option) = [option ; sampled_reward];
        history_discounted(option) = sampled_reward;
    end
    % --- Set starting weights / depth ---
    p.depth = softmax1(history(2,1:numWells),p.temp);
    d_history(:,1) = p.depth';
    
    % --- Initialise counters ---
    counter = 1+numWells;               % Current MAB step
    sample_count = zeros(1,numWells);   % How many times has opt been sampled
    
% ============================================================
%                   BEGIN FNS-MAB SIMULATION
% ============================================================
    for i = 1 : window : n          % Sim time separated into MAB windows   
        for w = i:i+window-1    % Execute fHMC simulation in window
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
        
        % --- Pick one point from history and return payoff ---
        chosen_point = X(:,w - floor(window/2))';
        chosen_option = wellCheck(chosen_point,p,payoffs(counter,:));
        
%         chosen_point = mvnrnd(p.location(chosen_option,:),p.sigma2(chosen_option)*Id,1);
        reward = generateReward(chosen_point,p,payoffs(counter,:));
        
        
        % --- Record reward and chosen option + applying discount ---
        reward_points(:,counter) = chosen_point;
        history(:,counter) = [chosen_option ; reward];
        history_discounted(counter) = history(2,counter);
        history_discounted(history(1,1:counter)==chosen_option) = ...
             history_discounted(history(1,1:counter)==chosen_option)*p.l;
%     
    % ============================================================
    %       Update beliefs w/ dUCB + softmax: apply to depth
    % ============================================================
            
        for opt = 1:numWells
            option_vec = (history(1,1:counter) == opt);
            rewards_disc = history_discounted(1:counter) .* option_vec;
            EV(opt) = mean(rewards_disc(rewards_disc~=0));
            sample_count(opt) = sum(option_vec);
        end
        IB = sqrt(2*log(counter) ./ sample_count);
        
        p.depth = softmax1(EV+p.n*IB,p.temp);
        d_history(:,counter) = p.depth';
        counter = counter + 1;  
    end
end



%% Functions 
function weights = softmax1(vec,temp)
    % Softmax function
    weights = exp(vec/temp)/sum(exp(vec/temp));
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

function reward = generateReward(coords,p,payoff)
    % Find payoff for each coordinate given the Gaussian parameters`
    % Should change to be an independent well vs the entire plane?
    reward = 0;
    for i = 1:length(p.sigma2)
        reward = reward + payoff(i)*mvnpdf(coords, ...
                p.location(i,:),p.sigma2(i)*[1,0;0,1]);
    end
end

function chosen = wellCheck(x,p,payoff)
    % Determines which well the point is sampling from.
    % Compare the gradient from each of the wells, pick largest magnitude.
    grad_array = zeros(size(payoff));
    
    for j = 1:length(payoff) % optimise: compute x,y stuff together
        distx = x(1)-p.location(j,1);
        disty = x(2)-p.location(j,2);
        stim = payoff(j) * exp(-0.5*(distx.^2+disty.^2)/p.sigma2(j));
        
        grad_array(j) = norm([stim.*(-distx/p.sigma2(j)),stim.*(-disty/p.sigma2(j))]);

    end
    [~,chosen] = max(grad_array);
        
end

function sdepth = smoothSwitching(depth,time)
    
end
