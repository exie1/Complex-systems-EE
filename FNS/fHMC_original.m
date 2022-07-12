function [X,t] = fHMC_original(T,a,p,avg)

    m = 2;
    dt = p.dt;%1e-3; %integration time step (s)
    dta = dt.^(1/a); %fractional integration step
    n = floor(T/dt); %number of samples
    t = (0:n-1)*dt;

    x = zeros(2,avg)+[1e-10;0]; %initial condition for each parallel sim
    v = zeros(2,avg)+[1;0];

    ca = gamma(a-1)/(gamma(a/2).^2);

    X = zeros(m,n,avg);
    for i = 1:n
        fx = 0;
        fy = 0;
        for j = 1:size(p.location,1)
%             pot = (x(1,:)/p.width(j) - p.location(j,1)).^2 + ... 
%                 (x(2,:)/p.width(j) - p.location(j,2)).^2 - p.depth(j);
%             gradx = 2*p.depth(j)/p.width(j) * (x(1,:)/p.width(j) - p.location(j,1));
%             grady = 2*p.depth(j)/p.width(j) * (x(2,:)/p.width(j) - p.location(j,2));
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

        X(:,i,:) = x;
    end
end

