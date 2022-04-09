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
obj.avg = 3;
obj.depth = [1,1,1,1];
obj.radius2 = [1,1,1,1];

obj.location = [-1,1;1,1;-1,-1;1,-1];

obj.rewardMu = [1,2,3,4];
obj.rewardSig = 0.5*obj.rewardMu;

tic
[X,scores,history] = obj.fHMC_MAB();
toc