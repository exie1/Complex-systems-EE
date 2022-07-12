function [mean_duration,total_duration] = closest_stim(X,p,ths,avg)
    % Find mean duration the sampling point stays in a stim

    mean_duration = zeros(size(p.location,1),avg);
    total_duration = zeros(size(p.location,1),avg);
    for j = 1:avg
        for i = 1:size(p.location,1)
            loc = p.location(i,:);
            dist_stim = sqrt((X(1,:,j)-loc(1)).^2 + (X(2,:,j)-loc(2)).^2);
            in_stim = [0,(dist_stim<ths(i)),0];
            exit_indices = find([false,in_stim]~=[in_stim,false]);
            in_stim_times = (exit_indices(2:2:end) - exit_indices(1:2:end-1))*p.dt;
            mean_duration(i,j) = mean(in_stim_times);
            total_duration(i,j) = sum(in_stim_times);
        end
    end
end

% Runs very fast, not much need to vectorise

% function mean_durations = closest_stim(X,p,ths)
%     end_pts = zeros(size(squeeze(X(1,1,:))));
%     mean_durations = zeros(size(p.location,1),1);
%     for i = 1:size(p.location,1)
%         loc = p.location(i,:);
%         dist_stim = squeeze(sqrt((X(1,:,:)-loc(1)).^2 + (X(2,:,:)-loc(2)).^2)).';
%         in_stim = [end_pts,(dist_stim<ths(i)),end_pts];
%         exit_indices = find([false,in_stim]~=[in_stim,false]);
%         in_stim_times = (exit_indices(2:2:end) - exit_indices(1:2:end-1))*p.dt;
%         mean_durations(i) = mean(in_stim_times);
%     end
% end
