
a = 1.2; % Levy tail exponent
p.sigma2 = 0.32; % sigma^2 where sigma is the width of the well
p.beta = 1; % beta coefficient
p.gam = 2; % strength of the Levy noise
p.dt = 1e-3; % integration time step
p.location = pi/2; %modal location 
p.depth1 = 1;
p.depth2 = 1;
avgs = 5; % number of simulations averaged over


MET_list = [];
param_list = linspace(1,10,15);
% logspace(log10(1),log10(10),15); 
% linspace(0.5,2,15)

for i = 1:length(param_list)
    p.depth1 = param_list(i);
    lst = [];
    for j = 1:avgs 
       [X,t,MET] = fns_sim_v5(a,p);
       lst = [lst, MET];
    end
    MET_list = [MET_list, mean(lst)];
    disp(p.depth1)
end
disp('Done')
%%
plot(param_list,MET_list,'.-')
xlabel('Stimulus depth')
ylabel('Mean duration in stim')



%%

plot(param_list,MET_list,'.-')
xlabel('Levy noise strength \gamma')
ylabel('Mean duration in stim')

log_param = log10(param_list(~isnan(MET_list)));
log_MET = log10(MET_list(~isnan(MET_list)));
% log_gam = log_gam(log_gam>1.5)

a = polyfit(log_param,log_MET,1);
hold on
plot(sig_list,10^a(2) * a_list.^a(1))

f_obs = 10^a(2) * a_list(~isnan(MET_list)).^a(1);
f_exp = MET_list(~isnan(MET_list));
sum((f_obs-f_exp).^2./f_exp)/(length(f_obs) - 2)

%MET - gamma strength: a = [-1.128590418152765,0.190831940982809]


%%

a = 1.2; % Levy tail exponent
p.sigma2 = 0.32; % sigma^2 where sigma is the width of the well
p.beta = 1; % beta coefficient
p.gam = 2; % strength of the Levy noise
p.dt = 1e-3; % integration time step
p.location = pi/2; %modal location 
p.depth1 = 1;
p.depth2 = 0;
avgs = 5; % number of simulations averaged over


MET_list1 = [];
MET_list2 = [];

param_list = linspace(1,100,10);
% logspace(log10(1),log10(10),15); 
% linspace(0.5,2,15)

for i = 1:length(param_list)
    p.depth1 = param_list(i);
    [X,t,exit_times] = fns_sim_v5(a,p);
    MET1 = mean(exit_times(1:2:end));
    MET2 = mean(exit_times(2:2:end));
    if X(1,1) > 0
        MET_list1 = [MET_list1, MET1];
        MET_list2 = [MET_list2, MET2];
    else
        MET_list1 = [MET_list1, MET2];
        MET_list2 = [MET_list2, MET1];
    end
%     disp(p.depth1)
end
disp('Done')


%%
hold on
plot(param_list,MET_list1)
plot(param_list,MET_list2)


