a = 1.3; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e2; % simulation time
num_parallel = 1;

%--------Defining stimuli and running simulation--------------------

% R = pi/2;
% theta = pi/2;   % for triangle stim
% p.location = [R*cos(theta),R*sin(theta); 
%     R*cos(theta+2*pi/3),R*sin(theta+2*pi/3);
%     R*cos(theta+4*pi/3),R*sin(theta+4*pi/3)];
% p.depth = [1,1,1];
% p.radius2 = [1,1,1].^2;

p.location = [-1,-1;1,1];     %[-1,1; 1, -1]*pi/2; 
p.depth = [1,1];% + zeros(length(p.location),1);
p.radius2 = [1,1].^2;% + zeros(length(p.location),1);
ths = sqrt(p.radius2);

tic
[X,t] = fHMC(T,a,p,num_parallel);
[mean_dur,total_dur] = closest_stim(X,p,ths,num_parallel);
toc

disp([mean(mean_dur,2),std(mean_dur,0,2),mean(total_dur,2),std(total_dur,0,2)])
disp(mean(total_dur,2)/T)

%% ---------------------Displaying results--------------------------
figure

subplot(1,2,1)
coords = makelattice(100);
plot3(coords(1,:),coords(2,:),quadratic_rewards(coords,p),'.','markersize',1) 
xlabel('x')
ylabel('y')

subplot(1,2,2)
% hist3(squeeze(X(:,:,1))',[50,50],'CDataMode','auto','FaceColor','interp')
% plot3(X(1,:,1),X(2,:,1),rewards(:,1)*1400,'.','markersize',1) 
histogram2(X(1,:,1),X(2,:,1),'Normalization','probability','FaceColor','flat')
xlabel('x')
ylabel('y')

% subplot(1,3,3)
% plot(X(1,:,1),X(2,:,1),'.','markersize',0.1)
% hold on
% viscircles(p.location, ths);
% xlim([-pi,pi])
% ylim([-pi,pi])
% xlabel('x')
% ylabel('y')

%% Looping business

avgs = 30; % should average more or increase sampling time for more stimuli
depth_list = logspace(-1,1,30);

total_timesD = [];
mean_timesD = [];
for i = 1:length(depth_list)
    p.depth(1) = depth_list(i);
    ths(1) = depth_list(i); 
    [X,t] = fHMC(1e3,a,p,avgs);
    [mean_dur,total_dur] = closest_stim(X,p,ths,avgs);
    total_timesD = [total_timesD, mean(total_dur,2)];
    mean_timesD = [mean_timesD, mean(mean_dur,2)];
    disp([mean(mean_dur,2),mean(total_dur,2)])
    disp(i)
end


%%

figure 
hold on

% plot(sqrt(depth_list),total_times(1,:))
% plot(sqrt(depth_list),total_times(2,:))
% plot(sqrt(depth_list),sum(total_times,1))
% xlabel('depth of stimuli 1')
% ylabel('total duration in stimuli (s)')
% ylim([0,100])
% title('Total duration in stimuli over 100 s simulation')
% legend('Stimulus 1 (varying depth)', 'Stimulus 2','Sum')

plot(depth_list,total_timesD(1,:))
plot(depth_list,total_timesD(2,:))
plot(depth_list,sum(total_timesD,1))
xlabel('depth of stimulus')
ylabel('total duration in stimuli')
title('Total duration in stimuli over 1000 * 1e-3 simulation')
legend('Stimulus 1 (varying depth)', 'Stimulus 2','Sum')

%%

function coords = makelattice(n)
    [x,y] = meshgrid(linspace(-pi,pi,n),linspace(-pi,pi,n));
    coords = [];
    for i = 1:size(x,1)^2
        coords = [coords, [x(i);y(i)]];
    end
end
