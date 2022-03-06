function [X,t] = fHMC_2d(T,a,p)
    if isempty(p)    
        p.location = pi/2;
        p.sigma2 = 1;
        p.gam = 2;
        p.dt = 1e-3;
        p.beta = 1;
        p.depth1 = 1;
        p.depth2 = 1;
    end

    m = 2;
    dt = p.dt;%1e-3; %integration time step (s)
    dta = dt.^(1/a); %fractional integration step

    n = floor(T/dt); %number of samples
    t = (0:n-1)*dt;
    x = [0;0];%p.x0; %initial condition
    v = [1;0];
    X = zeros(m,n);

    t0 = tic;
    tic

    ca = gamma(a-1)/(gamma(a/2).^2);

    for i = 1:n
        % non-fractional grad of 2d gaussian wells
        p1 = p.depth1 * exp(-0.5*(x(1)-p.location).^2/p.sigma2 - 0.5*x(2).^2/p.sigma2 );
        p2 = p.depth2 * exp(-0.5*(x(1)+p.location).^2/p.sigma2 - 0.5*x(2).^2/p.sigma2 );

        fx = p1*(-(x(1)-p.location)/p.sigma2 ) + p2*(-(x(1)+p.location)/p.sigma2 );
        fy = -x(2)/p.sigma2;
        f = [fx/(p1+p2); fy];

        dL = stblrnd(a,0,p.gam,0,[2 1]);
        r = sqrt(sum(dL.*dL)); %step length

        %g = [g1 ; g2];
        th = rand*2*pi;
        g = r*[cos(th);sin(th)];

        vnew = v + p.beta*ca*f*dt;
        xnew = x + p.gam*ca*f*dt + p.beta*v*dt + g*dta;

        x = xnew;
        v = vnew;

        x = wrapToPi(x); % apply periodic boundary to avoid run-away

        if toc -t0 > 120
            disp('Time out!')
            %return
        end

        X(:,i) = x;
    end
end