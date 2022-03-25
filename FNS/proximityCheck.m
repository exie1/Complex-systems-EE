function option = proximityCheck(points,centres)
    % Find which centre each point is closest to
    % Locations in format [x1,y1 ; x2,y2; ...]
    % Points in format [x1,x2,x3,... ; y1,y2,y3,...]
    prox = zeros(size(centres,1),size(points,2));
    for i = 1:size(centres,1)
        centre = centres(i,:)';
        prox(i,:) = vecnorm(points-centre);
    end
    [~,option] = min(prox);
end