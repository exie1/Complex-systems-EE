function reward = quadratic_rewards(X,p)
    % What reward does each sampling point attain for a quadratic potential
    if size(X,3) == 1
        reward = zeros(1,size(X,2));
    else
        reward = zeros(size(X,2),size(X,3));
    end
    
    for j = 1:size(p.location,1)
        z = p.depth(j)*(((X(1,:,:) - p.location(j,1)).^2 + ...
                (X(2,:,:) - p.location(j,2)).^2)/p.radius2(j) - 1);
        reward = reward + squeeze(z.*(z<=0)) ;        
    end
end