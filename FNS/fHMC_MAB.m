function [X,t,history] = fHMC_MAB(T,a,p,window,avg)
    % Same fHMC algorithm, but optimised for multi-armed bandit
    % simulations. Change width for bottom-up attention, change depth for
    % top-down attention, change Levy noise for random exploration.

    m = 2;           % dimensions
    dt = p.dt;%1e-3; % integration time step (s)
    dta = dt.^(1/a); % fractional integration step
    n = floor(T/dt); % number of samples
    t = (0:n-1)*dt;  % time
    max_rad = vecnorm(p.location(1,:)-p.location(2,:))/2;

    x = zeros(2,avg)+[1e-10;0]; % initial condition for each parallel sim
    v = zeros(2,avg)+[1;0];     % ^ but velocity

    ca = gamma(a-1)/(gamma(a/2).^2);    % fractional derivative approx

    expectation = zeros(1,length(p.rewardMu));
    history = zeros(2,round(n/window)); 
    X = zeros(m,n,avg);
    counter = 1;
    for i = 1:window:n      % num steps separated into time windows    
        for w = i:i+window-1
            fx = 0;
            fy = 0;     % FRACTIONAL DERIVATIVE CALCULATION
            for j = 1:size(p.location,1)
                pot = p.depth(j)*(((x(1,:) - p.location(j,1)).^2 + ...
                    (x(2,:) - p.location(j,2)).^2)/p.radius2(j) - 1);
                gradx = (x(1,:)-p.location(j,1))*2*p.depth(j)/p.radius2(j);
                grady = (x(2,:)-p.location(j,2))*2*p.depth(j)/p.radius2(j);
                fx = fx + gradx.*(pot<=0);
                fy = fy + grady.*(pot<=0);
            end
            f = -[fx;fy];

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
            X(:,w,:) = x;    % record position
        end
        chosen = proximityCheck(X(:,i:w,:),p.location);
        history(1,counter) = chosen;
        history(2,counter) = p.rewardSig(chosen)*randn()+p.rewardMu(chosen);
        counter = counter + 1;
        
        
        for opt = 1:size(p.rewardMu)
            rewards = history(2,:) .* (history(1,:) == opt);
            expectation(opt) = mean(rewards(rewards~=0));
        end
        weights = exp(expectation)./sum(exp(expectation));
        p.radius2 = (max_rad*weights).^2;
    end
end



% Different optimisations?


%     for i = 1:round(n/window)      % num steps / time window
%         MAB_list = zeros(2,window,avg);
%         for j = 1:window    % find closest reward
%             fx = 0;
%             fy = 0;
%             for j = 1:size(p.location,1)
%                 pot = p.depth(j)*(((x(1,:) - p.location(j,1)).^2 + ...
%                     (x(2,:) - p.location(j,2)).^2)/p.radius2(j) - 1);
%                 gradx = (x(1,:)-p.location(j,1))*2*p.depth(j)/p.radius2(j);
%                 grady = (x(2,:)-p.location(j,2))*2*p.depth(j)/p.radius2(j);
%                 fx = fx + gradx.*(pot<=0);
%                 fy = fy + grady.*(pot<=0);
%             end
%             f = -[fx;fy];
% 
%             dL = stblrnd(a,0,p.gam,0,[2,avg]); 
%             r = sqrt(sum(dL.*dL,1)); %step length
% 
%             th = rand(1,avg)*2*pi;
%             g = r.*[cos(th);sin(th)];
% 
%             % Stochastic fractional Hamiltonian Monte Carlo
%             vnew = v + p.beta*ca*f*dt;
%             xnew = x + p.gam*ca*f*dt + p.beta*v*dt + g*dta;
% 
% 
%             x = xnew;
%             v = vnew;
% 
%             x = wrapToPi(x); % apply periodic boundary to avoid run-away
% 
%             MAB_list(:,j,:) = x;
%             X(:,i+j-1,:) = x;
% %             X(:,i+j-1,:) = x;
%         end
%         history(i) = proximityCheck(MAB_list,p.location);
%     end


%     for i = 1:n      % num steps / time window
%         
%         fx = 0;
%         fy = 0;
%         for j = 1:size(p.location,1)
%             pot = p.depth(j)*(((x(1,:) - p.location(j,1)).^2 + ...
%                 (x(2,:) - p.location(j,2)).^2)/p.radius2(j) - 1);
%             gradx = (x(1,:)-p.location(j,1))*2*p.depth(j)/p.radius2(j);
%             grady = (x(2,:)-p.location(j,2))*2*p.depth(j)/p.radius2(j);
%             fx = fx + gradx.*(pot<=0);
%             fy = fy + grady.*(pot<=0);
%         end
%         f = -[fx;fy];
% 
%         dL = stblrnd(a,0,p.gam,0,[2,avg]); 
%         r = sqrt(sum(dL.*dL,1)); %step length
% 
%         th = rand(1,avg)*2*pi;
%         g = r.*[cos(th);sin(th)];
% 
%         % Stochastic fractional Hamiltonian Monte Carlo
%         vnew = v + p.beta*ca*f*dt;
%         xnew = x + p.gam*ca*f*dt + p.beta*v*dt + g*dta;
% 
% 
%         x = xnew;
%         v = vnew;
% 
%         x = wrapToPi(x); % apply periodic boundary to avoid run-away
% 
%         X(:,i,:) = x;
%         if mod(i,window) == 0
%             history(i/window) = proximityCheck(X(:,i+1-window:i,:),p.location);
%         end
%     end

