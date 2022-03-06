function reward = gaussian_rewards(X,p)
    % What reward does each sampling point attain?
    % Right now reward is just the gaussian lattice, aiming to find the
    % 'deepest' stimuli (multipler = x1 for now)
    if size(X,3) == 1
        reward = zeros(1,size(X,2));
    else
        reward = zeros(size(X,2),size(X,3));
    end
    
    for j = 1:size(p.location,1)
        reward = reward + p.depth(j)*squeeze(exp(-0.5*((X(1,:,:)-p.location(j,1)).^2 ...
            +(X(2,:,:)-p.location(j,2)).^2)/p.sigma2(j)));        
    end
end