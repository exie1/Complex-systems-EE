function option = proximityCheck(points,centres)
    % Find which centre each point is closest to
    % Locations in format [x1,y1 ; x2,y2; ...]
    % Points in format [x1,x2,x3,... ; y1,y2,y3,...] + avg in 3D
    % Returns row vector of closest point for each trial
    if size(points,3) == 1
        prox = zeros(size(centres,1),size(points,2));
        for i = 1:size(centres,1)
            centre = centres(i,:)';
            prox(i,:) = vecnorm(points-centre);
        end
        [~,closest] = min(prox);
        option = mode(closest);
    else 
        prox = zeros(size(centres,1),size(points,2),size(points,3));
        for i = 1:size(centres,1)
            centre = centres(i,:)';
            absdiff = points(:,:,:) - centre;
            prox(i,:,:) = squeeze(absdiff(1,:,:).^2 + absdiff(2,:,:).^2);
        end
        [~,closest] = min(prox,[],1);
        option = mode(squeeze(closest),1);
    end
end