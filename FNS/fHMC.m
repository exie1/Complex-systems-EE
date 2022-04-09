classdef fHMC
    properties
        T = 1e3              % Total simulation time
        a = 1.3              % Levy noise tail exponent
        gamma = 1          % Levy noise strength coefficient
        beta = 1           % Momentum coefficient
        tau = 1            % Softmax temperature
        dt = 1e-3             % Timestep
        location        % Location of each well [x1,y1 ; x2,y2 ; ...]
        depth           % Depth of each well
        radius2         % Radius of each well
        rewardMu        % Reward corresponding to each well
        rewardSig       % Variance corresponding to each well 
        mabStep         % Number of multi-armed bandit steps
        maxRad = pi
        avg             % Number of averages
    end
    methods
        % Apply softmax function on matrix rows
        function weights = softmax1(fHMC,vec)
            % Rowwise softmax function
            weights = exp(vec/fHMC.tau)./sum(exp(vec/fHMC.tau),2);
        end
        
        function option = proximityCheck(fHMC,vec)
            prox = zeros(size(fHMC.location,1),size(vec,2),size(vec,3));
            for i = 1:size(fHMC.location,1)
                centre = fHMC.location(i,:)';
                prox(i,:,:) = squeeze(vecnorm(vec-centre));
            end
            [~,closest] = min(prox,[],1);
            if size(vec,3) == 1
                option = mode(closest);
            else
                option = mode(squeeze(closest),1);
            end
        end
        
        % Calculate fractional derivative of potential
        function f = getPotential(fHMC,x)
            fx = 0;
            fy = 0;     
            fn = 0;
            % Quadratic potential
            if isprop(fHMC,'radius2')
                for j = 1:size(fHMC.location,1)
                    pot = fHMC.depth(:,j).*(((x(:,1) - fHMC.location(j,1)).^2 + ...
                        (x(:,2) - fHMC.location(j,2)).^2)./fHMC.radius2(:,j) - 1);
                    gradx = (x(:,1)-fHMC.location(j,1))*2.*fHMC.depth(:,j)./fHMC.radius2(:,j);
                    grady = (x(:,2)-fHMC.location(j,2))*2.*fHMC.depth(:,j)./fHMC.radius2(:,j);
                    fx = fx + gradx.*(pot<=0);
                    fy = fy + grady.*(pot<=0);
                end
                f = -[fx,fy];
            % Gaussian potential
            elseif isprop(fHMC,'sigma2')
                for j = 1:size(fHMC.location,1) % optimise: compute x,y stuff together
                    distx = x(1,:)-fHMC.location(j,1);
                    disty = x(2,:)-fHMC.location(j,2);
                    stim = fHMC.depth(j) * exp(-0.5*(distx.^2+disty.^2)/fHMC.sigma2(j));
                    
                    fx = fx + stim.*(-distx/fHMC.sigma2(j));
                    fy = fy + stim.*(-disty/fHMC.sigma2(j));
                    fn = fn + stim;
                end
                f = [fx; fy]./fn;  % log derivative: add e-15 to avoid 0/0.
            end
        end 
        
        % Apply sFHMC step on x vector (+ averages)
        function x = sFHMC(fHMC,x,v,ca)
            f = fHMC.getPotential(x);

            dL = stblrnd(fHMC.a,0,fHMC.gamma,0,[fHMC.avg,2]); 
            r = sqrt(sum(dL.*dL,2)); %step length
            th = rand(fHMC.avg,1)*2*pi;
            g = r.*[cos(th),sin(th)];

            % Stochastic fractional Hamiltonian Monte Carlo
            vnew = v + fHMC.beta*ca*f*fHMC.dt;
            xnew = x + fHMC.gamma*ca*f*fHMC.dt + fHMC.beta*v*fHMC.dt + g*fHMC.dt.^(1/fHMC.a); 
            x = xnew;
            v = vnew;

            x = wrapToPi(x); % apply periodic boundary to avoid run-away
        end
       
        % Multi-armed bandit FHMC simulation
        function [X,scores,history] = fHMC_MAB(fHMC)
            numWells = length(fHMC.rewardMu);
            window = fHMC.T/fHMC.dt/fHMC.mabStep;
            avg = fHMC.avg;
            n = floor(fHMC.T/fHMC.dt);
            
            x = zeros(avg,2) + [0,0];
            v = zeros(avg,2) + [1,0];       % consider changing to 0,0
            X = zeros(2,n,avg);
            ca = gamma(fHMC.a-1)/(gamma(fHMC.a/2).^2);
            
            expectation = zeros(avg,numWells);
            scores = zeros(avg,round(n/window)+numWells);
            history = zeros(avg,round(n/window)+numWells);
            
            history(:,1:numWells) = zeros(avg,1)+(1:numWells);
            scores(:,1:numWells) = [fHMC.rewardSig.*randn(avg,numWells)+fHMC.rewardMu];
            weights = fHMC.softmax1(scores(:,1:numWells));
            fHMC.radius2 = (fHMC.maxRad*weights).^2;
            fHMC.depth = fHMC.radius2;
            
            counter = 1 + numWells;
            for i = 1:window:n
                for w = i:i+window-1
                    x = fHMC.sFHMC(x,v,ca);
                    X(:,w,:) = x';    
                end
                chosen = fHMC.proximityCheck(X(:,i:w));
                history(:,counter) = chosen';
                scores(:,counter) = fHMC.rewardSig(chosen)'.*randn(avg,1)+fHMC.rewardMu(chosen)';
                
                for opt = 1:numWells
                    rewards = scores(:,1:counter).*(history(:,1:counter)==opt);
                    expectation(:,opt) = sum(rewards,2)./sum(logical(rewards),2);
                    mean(rewards(rewards~=0),2);
                end
                weights = fHMC.softmax1(expectation);
                fHMC.radius2 = (fHMC.maxRad*weights).^2;
                fHMC.depth = fHMC.radius2;
                counter = counter + 1;
            end  
        end
        
    end
end

%         chosen = proximityCheck(X(:,i:w),p.location);
%         history(1,counter) = chosen;
%         history(2,counter) = p.rewardSig(chosen)*randn()+p.rewardMu(chosen);
%         history_rad(:,counter) = p.radius2';
%         
%         % Updating well parameters according to sampled history
%         for opt = 1:length(p.rewardMu)
%             rewards = history(2,1:counter) .* (history(1,1:counter) == opt);
%             expectation(opt) = mean(rewards(rewards~=0));
%         end
%         weights = softmax1(expectation);
%         p.radius2 = (max_rad*weights).^2;
%         p.depth = p.radius2; %sqrt or no?
%         counter = counter + 1;
%     end
% end