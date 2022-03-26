a = 1.3; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e3; % simulation time
MAB_steps = 500;
window = T/p.dt/MAB_steps;
num_parallel = 1;

%--------Defining stimuli and running simulation--------------------

R = pi/2;
theta = pi/2;   % for triangle stim
p.location = [R*cos(theta),R*sin(theta); 
    R*cos(theta+2*pi/3),R*sin(theta+2*pi/3);
    R*cos(theta+4*pi/3),R*sin(theta+4*pi/3)];
p.depth = [1,1,1];

p.radius2 = [1,1,1].^2;
p.rewardMu = [3,4,5];       % should initialise this as sampled values
p.rewardSig = [5,4,3]/2;

tic
[X,t,history] = fHMC_MAB(T,a,p,window,num_parallel);
toc

regret = 1 - (sum(history(2,:))/max(p.rewardMu)*MAB_steps);
%%
figure
subplot(1,2,1)
plot(history(1,:))
xlabel('Step')
ylabel('Chosen option')

subplot(1,2,2)
histogram2(X(1,:,1),X(2,:,1),'Normalization','probability','FaceColor','flat')
xlabel('x')
ylabel('y')

%%

figure
oneOpt = history(2,:) .* (history(1,:) == 3);
histogram(oneOpt)
(5*500 - sum(history(2,:)))/(5*500)