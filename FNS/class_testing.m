obj = fHMC();

MAB_steps = 500;
window = T/obj.dt/MAB_steps;
num_parallel = 1;

obj.a = 1.3; % Levy tail exponent
obj.beta = 1; % beta coefficient
obj.gamma = 1; % strength of the Levy noise
obj.dt = 1e-3; % integration time step
obj.tau = 1; % softmax sensitivity
obj.T = 1e3; % simulation time
obj.mabStep = 500;
obj.avg = 1;
% obj.depth = [1,1,1,1];
% obj.radius2 = [1,1,1,1];
% obj.location = [-1,1;1,1;-1,-1;1,-1];
R = pi/2;
theta = pi/2;   % for triangle stim
obj.depth = [1,1,1];
obj.radius2 = [1,1,1].^2;
obj.location = [R*cos(theta),R*sin(theta); 
    R*cos(theta+2*pi/3),R*sin(theta+2*pi/3);
    R*cos(theta+4*pi/3),R*sin(theta+4*pi/3)];


obj.rewardMu = 2 + 3*(1:3);
obj.rewardSig = 0.5*obj.rewardMu;
% obj.rewardMu = [1,2,3,4];
% obj.rewardSig = 0.5*obj.rewardMu;

tic
[X,scores,history] = obj.fHMC_MAB();
toc

%%
trial = 1;

figure
subplot(1,2,1)
plot(history(trial,:))
xlabel('Step')
ylabel('Chosen option')
title('Choice history')
xlim([0,obj.mabStep])

subplot(1,2,2)
histogram2(X(1,:,trial),X(2,:,trial),'Normalization','probability','FaceColor','flat')
xlabel('x')
ylabel('y')
title('Sampled distribution')