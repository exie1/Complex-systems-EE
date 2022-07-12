function [option,closest] = proximityCheck(history,centres)
    % Find which centre each point is closest to
    % Locations in format [x1,y1 ; x2,y2; ...]
    % Points in format 2 x time x avg
    % Returns row vector of closest point for each trial
    prox = zeros(size(centres,1),size(history,2),size(history,3));
    for i = 1:size(centres,1)
        centre = centres(i,:)';
        prox(i,:,:) = squeeze(vecnorm(history-centre));
    end
    [~,closest] = min(prox,[],1);
    if size(history,3) == 1
        option = mode(closest);
    else
        option = mode(squeeze(closest),1);
    end
end