%% Problem setup
payoffs = csvread('payoffs\payoffs_step2.csv')';
% payoffs = payoffs(:,2:3);
% p.location = [-1,0;1,0];
% p.sigma2 = 0.4 * [1,1];
% p.depth = [1,1];
% p.rewardMu = payoffs(1,:);
% p.rewardSig = zeros(1,2) + 4;

payoffs = payoffs(:,3);
p.location = [-1,0;1,0];
p.sigma2 = 0.4 * [1,1];
p.depth = [1,1];
p.rewardMu = payoffs(1,:);
p.rewardSig = zeros(1,1) + 4;



p.dt = 1e-3; % integration time step
MAB_steps = 300;
% p.maxVal_d = 1; % Value here cancels out in gradient calc

p.a = 1.1;      % Levy tail exponent
p.gam = 0;      % strength of the Levy noise
p.beta = 1;     % momentum term: amount of acceleration
p.maxVal_d = 0.4;

p.temp = 1;     % softmax temperature
p.l = 0.99;       % recency bias
p.T = 0.9e2;      % simulation time: integer multiple of MAB_steps pls
p.n = 0.9;        % Amount of exploitation - directed exploration

tic
[X,t,history,history_rad] = fHMC_dynMABGaussian(p,payoffs,MAB_steps);
toc

optimal = sum(max(payoffs,[],2));
regret = 1 - (sum(history(2,:))/optimal);
[cnt_unique, uniq] = hist(history(1,:),unique(history(1,:)));
disp('Proportion of samples + overall regret')
disp([cnt_unique/sum(cnt_unique),regret])

%% Plotting spatial walk
figure
subplot(1,2,1)
plot(X(1,:),X(2,:),'.','markerSize',0.01)
axis square
subplot(1,2,2)
plot(t,X(2,:),'lineWidth',0.1)
% hist3(X')


%% Fourier transform
y = sin(t) + 2*randn(1,length(t));

Y = fft(y);
L = length(y);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
fs = 1/p.dt;        % Sampling frequency
f = (0:L/2)*fs/L;

figure
subplot(2,1,1)
plot(t,y)
subplot(2,1,2)
plot(f,P1)

%%
period = p.dt;
fs = 1/period;
data = X(1,:);
Y = fft(data);
n = length(data);
f = (0:n-1)*fs/n;
power = abs(Y).^2/n;

plot(f,power)

