%% Defining parameters
a = 1.5; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e3; % simulation time
num_avg = 2;

p.location = [-1,0;1,0]; %[pi/2,0 ; -pi/2,0 ; 0,pi/2 ; 0,-pi/2];    
p.sigma2 = [0.3,0.3];    %[0.32,0.32,0.32,0.32]; 
p.depth = [1,1];        %[1,1,1,1];
ths = 2*sqrt(p.sigma2);

tic
[X,t] = fHMC_opt(T,a,p,num_avg);
toc
% durations = closest_stim(X,p,ths,num_avg);
% disp(durations)
%% Plotting

figure('color','w');
plot(X(1,:,1),X(2,:,1),'.','markersize',1)
hold on
viscircles(p.location, ths);
xlabel('x')
ylabel('y')
axis equal

%% Power spectrum
figure('color','w');
subplot(2,1,1)
sliding_window = 1;
x = movmean(X(1,:,1),sliding_window);
plot(t(:),x,'markersize',1)
hold on
xlabel('t')
ylabel('x')

subplot(2,1,2)
fs = 1/p.dt;
Y = fft(x);
n = length(x);
f = (0:n-1)*fs/n;
power = abs(Y).^2/n;

plot(f,power)
xlabel('Frequency (Hz)')
% xlim([0,1])


%% Step distribution
jumps = sqrt(diff(X(1,:)).^2 + diff(X(2,:)).^2);

numBins = 50;
[binCenters,Nnorm] = binLogLog(numBins,jumps);
binCenters = binCenters(Nnorm~=0);
Nnorm = Nnorm(Nnorm~=0);

figure('color','w');
plot(binCenters,Nnorm)
title('Log-log distribution of sampling step sizes')
xlabel('Step size')
ylabel('Frequency')

hold on
boundaries = [0.02,3];
segment = (binCenters>boundaries(1) & binCenters<boundaries(2));
Nnorm_cut = Nnorm(segment);
binCenters_cut = binCenters(segment);
index = polyfit(log10(binCenters_cut),log10(Nnorm_cut),1);
loglog(binCenters_cut, 10^index(2) * binCenters_cut.^index(1));
disp(index)


%% Looping business

sigma_list = logspace(log10(0.1),log10(3),20);
avgs = 30;
times = [];
for i = 1:length(depth_list)
    p.sigma2 = sigma_list(i);
%     ths = 2*sqrt(p.sigma2);
    [X,t] = fHMC_opt(T,a,p,avgs);
    durations = closest_stim(X,p,ths,avgs);
    times = [times, mean(durations)];
    disp([mean(durations),i])
end

%%
figure
plot(sigma_list,times)
xlabel('Stimulus variance')
ylabel('Mean duration in stim')
title('With max 2 SD (3) thresholding')


