[x,y] = meshgrid(linspace(-pi,pi,50),linspace(-pi,pi,50));
coords = [];
for i = 1:size(x,1)^2
    coords = [coords, [x(i);y(i)]];
end
location = [-pi/2,0;pi/2,0];
width = [1,2];
depth = [1,1];

fx = 0;
fy = 0;
fn = 0;
stim_hist = 0;
for j = 1:length(width)
    stim = depth(j)*exp(-0.5*((coords(1,:)-location(j,1)).^2+...
            (coords(2,:)-location(j,2)).^2)/width(j));
    fx = fx + stim.*(-(coords(1,:)-location(j,1))/width(j));
    fy = fy + stim.*(-(coords(2,:)-location(j,2))/width(j));
    fn = fn + stim;
    stim_hist = stim_hist + stim;
end
f = [fx;fy]./fn;


figure
subplot(1,3,1)
plot3(coords(1,:),coords(2,:),stim_hist)
title('Potential well')

subplot(1,3,2)
plot3(coords(1,:),coords(2,:),fx)
title('Gradient of potential')

subplot(1,3,3)
plot3(coords(1,:),coords(2,:),fx./fn)
title('Normalised?')

sgtitle(['width: ', num2str(width),', depth: ', num2str(depth)])


%% Assigment lol
eps = 8.85*10^-12;
9.109*10^-31;
c = 3*10^8;
g = 10^5;
e = 1.602*10^-19;
me = 9.109*10^-31;
B = 1*10^-9;
6*pi*eps*me^3*c^3/(e^4*g*B^2) / (60*60*24*365)


