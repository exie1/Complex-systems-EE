function option = proximityCheck(points,centres)
    % Find which centre each point is closest to
    % Locations in format [x1,y1 ; x2,y2; ...]
    % Points in format [x1,x2,x3,... ; y1,y2,y3,... ; av1,av2,...]
    
    if size(points,3) == 1
            prox = zeros(size(centres,1),size(points,2));
        for i = 1:size(centres,1)       % Looping over location of each well
            centre = centres(i,:)';
            prox(i,:,:) = vecnorm(points-centre);
        end
        [~,closest] = min(prox);
        option = mode(squeeze(closest));
    else
        prox = zeros(size(centres,1),size(points,2),size(points,3));
        for i = 1:size(centres,1)       % Looping over location of each well
            centre = centres(i,:)';
            prox(i,:,:) = vecnorm(points-centre,2,1);
        end
        [~,closest] = min(prox,[],1);
        option = mode(squeeze(closest),1);
    end
end