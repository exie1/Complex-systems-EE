classdef fHMC
    properties
        T = 1e3              % Total simulation time
        a = 1.3              % Levy noise tail exponent
        gamma = 1          % Levy noise strength coefficient
        beta = 1           % Momentum coefficient
        dt = 1e3             % Timestep
        location        % Location of each well [x1,y1 ; x2,y2 ; ...]
        depth           % Depth of each well
        radius2         % Radius of each well
        rewardMu        % Reward corresponding to each well
        rewardSig       % Variance corresponding to each well 
        mabStep         % Number of multi-armed bandit steps
        mabWindow       % 
        avg             % Number of averages
    end
    methods
       
        % Apply softmax function on matrix rows
        function weights = softmax1(vec,tau)
            % Rowwise softmax function
            weights = exp(vec/tau)./sum(exp(vec/tau),2);
        end
        
        % Calculate fractional derivative of potential
        function f = getPotential(fHMC,x)
            fx = 0;
            fy = 0;     
            fn = 0;
            % Quadratic potential
            if isfield(p,'radius2')
                for j = 1:size(fHMC.location,1)
                    pot = p.depth(:,j).*(((x(:,1) - p.location(j,1)).^2 + ...
                        (x(:,2) - p.location(j,2)).^2)./p.radius2(:,j) - 1);
                    gradx = (x(:,1)-p.location(j,1))*2.*p.depth(:,j)./p.radius2(:,j);
                    grady = (x(:,2)-p.location(j,2))*2.*p.depth(:,j)./p.radius2(:,j);
                    fx = fx + gradx.*(pot<=0);
                    fy = fy + grady.*(pot<=0);
                end
                f = -[fx,fy];
            % Gaussian potential
            elseif isfield(p,'sigma2')
                for j = 1:size(p.location,1) % optimise: compute x,y stuff together
                    distx = x(1,:)-p.location(j,1);
                    disty = x(2,:)-p.location(j,2);
                    stim = p.depth(j) * exp(-0.5*(distx.^2+disty.^2)/p.sigma2(j));
                    
                    fx = fx + stim.*(-distx/p.sigma2(j));
                    fy = fy + stim.*(-disty/p.sigma2(j));
                    fn = fn + stim;
                end
                f = [fx; fy]./fn;  % log derivative: add e-15 to avoid 0/0.
            end
        end 
        
        % Apply sFHMC step on x vector (+ averages)
        function x = sFHMC(fHMC,x)
            f = getPotential(x,p);

            dL = stblrnd(fHMC.a,0,fHMC.gamma,0,[fHMC.avg,2]); 
            r = sqrt(sum(dL.*dL,2)); %step length
            th = rand(fHMC.avg,1)*2*pi;
            g = r.*[cos(th),sin(th)];

            % Stochastic fractional Hamiltonian Monte Carlo
            vnew = v + fHMC.beta*ca*f*fHMC.dt;
            xnew = x + fHMC.gam*ca*f*fHMC.dt + fHMC.beta*v*fHMC.dt + g*dta; % change dta
            x = xnew;
            v = vnew;

            x = wrapToPi(x); % apply periodic boundary to avoid run-away
        end
       
        % Multi-armed bandit FHMC simulation
         
        
        
    end

end