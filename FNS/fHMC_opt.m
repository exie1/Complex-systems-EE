function [X,t] = fHMC_opt(p,avg)

%     p.location = rad * [-1,0;1,0];      % For parfor sim
%     p.sigma2 = sig2*[1,1];
    
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
        f = makeGaussian(x,p);

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
        X(:,i,:) = x;
    end
end


function f = makeGaussian(x,p)
    fx = 0;
    fy = 0;
    fn = 0;
    for j = 1:size(p.location,1) % optimise: compute x,y stuff together
        distx = x(1,:)-p.location(j,1);
        disty = x(2,:)-p.location(j,2);
        stim = p.depth(j) * exp(-0.5*(distx.^2+disty.^2)/p.sigma2(j))/sum(p.depth);

        fx = fx + stim.*(-distx/p.sigma2(j));
        fy = fy + stim.*(-disty/p.sigma2(j));
        fn = fn + stim;

    end
    f = [fx; fy]./fn;  % log derivative: add e-15 to avoid 0/0.

   
end

