a = 1.3; % Levy tail exponent
p.beta = 1; % beta coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e2; % simulation time
num_parallel = 30;

%--------Defining stimuli and running simulation--------------------
R = pi/2;
theta = pi/2;
% p.location = [R*cos(theta),R*sin(theta); 
%     R*cos(theta+2*pi/3),R*sin(theta+2*pi/3);
%     R*cos(theta+4*pi/3),R*sin(theta+4*pi/3)];
% p.depth = [1,1,1];
% p.sigma2 = ([1,1,1]/2).^2;
boundaries = linspace(-3*pi/5,3*pi/5,3);
p.location = build_lattice(boundaries); %[x1,y1 ; x2,y2 ; ...]
p.depth = 1 + zeros(length(p.location),2);
p.width = 1 + zeros(length(p.location),2);
ths = sqrt(p.depth).*p.width;

tic
[X,t] = fHMC(T,a,p,num_parallel);
toc

%% ---------------------Displaying results--------------------------
figure

subplot(1,2,1)
hist3(squeeze(X(:,:,1))',[50,50],'CDataMode','auto','FaceColor','interp')
% plot3(X(1,:,1),X(2,:,1),rewards(:,1)*1400,'.','markersize',1) 
xlabel('x')
ylabel('y')

subplot(1,2,2)
plot(X(1,:,1),X(2,:,1),'.','markersize',1)
hold on
% viscircles(p.location, ths);
xlim([-pi,pi])
ylim([-pi,pi])
xlabel('x')
ylabel('y')

