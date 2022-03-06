function [X,t] = fHMC_general(T,a,p)
    if isempty(p)    
        p.location = [pi/2,0;-pi/2,0];
        p.sigma2 = 0.32+zeroes(size(p.location,1));
        p.depth = 1+zeroes(size(p.location,1));
        p.gam = 2;
        p.dt = 1e-3;
        p.beta = 1;
    end

    m = 2;
    dt = p.dt;%1e-3; %integration time step (s)
    dta = dt.^(1/a); %fractional integration step
    n = floor(T/dt); %number of samples
    t = (0:n-1)*dt;

    x = [0;0];%p.x0; %initial condition
    v = [1;0];

    ca = gamma(a-1)/(gamma(a/2).^2);

    X = zeros(m,n);
    for i = 1:n
        fx = 0;
        fy = 0;
        fn = 0;
        for j = 1:size(p.location,1)
            stim = p.depth(j) * exp(-0.5*((x(1)-p.location(j,1)).^2 ...
                +(x(2)-p.location(j,2)).^2)/p.sigma2(j));
            fx = fx + stim*(-(x(1)-p.location(j,1))/p.sigma2(j));
            fy = fy + stim*(-(x(2)-p.location(j,2))/p.sigma2(j));
            fn = fn + stim;
        end

        f = [fx; fy]/fn;

        dL = stblrnd(a,0,p.gam,0,[2 1]);
        r = sqrt(sum(dL.*dL)); %step length

        th = rand*2*pi;
        g = r*[cos(th);sin(th)];

        vnew = v + p.beta*ca*f*dt;
        xnew = x + p.gam*ca*f*dt + p.beta*v*dt + g*dta;

        x = xnew;
        v = vnew;

        x = wrapToPi(x); % apply periodic boundary to avoid run-away

        X(:,i) = x;
    end
end
