classdef fHMC_class
    properties
        T               % Total simulation time
        a               % Levy noise tail exponent
        gamma           % Levy noise strength coefficient
        beta            % Momentum coefficient
        dt              % Timestep
        location        % Location of each well [x1,y1 ; x2,y2 ; ...]
        depth           % Depth of each well
        radius2         % Radius of each well
        rewardMu        % Reward corresponding to each well
        rewardSig       % Variance corresponding to each well 
    end
    methods
        function [X,t,history,history_rad] = fHMC_MAB(T,a,p,window,num_parallel)
            
        end
        function 
            
        end
    end
end