[x,y] = meshgrid(linspace(-pi,pi,100),linspace(-pi,pi,100));
coords = [];
for i = 1:size(x,1)^2
    coords = [coords, [x(i);y(i)]];
end
location = [-1,1;1,1;-1,-1;1,-1];
radius = [1,1,1,1]*0.3;
depth = [1,10,20,30];

fx = 0;
fy = 0;
fn = 0;
for j = 1:length(radius)
    stim = depth(j)*exp(-0.5*((coords(1,:)-location(j,1)).^2+...
            (coords(2,:)-location(j,2)).^2)/radius(j));
%     stim = (coords(1,:)/radius(j) + location(j,1)).^2 + ...
%         (coords(2,:)/radius(j) + location(j,2)).^2 - depth(j);
    fx = fx + stim.*(-(coords(1,:)-location(j,1))/radius(j));
    fy = fy + stim.*(-(coords(2,:)-location(j,2))/radius(j));
    fn = fn + stim;
end
f = [fx;fy]./fn;


figure
subplot(1,3,1)
plot3(coords(1,:),coords(2,:),fn)
title('Potential well')

subplot(1,3,2)
plot3(coords(1,:),coords(2,:),fx)
title('Gradient of potential')

subplot(1,3,3)
plot3(coords(1,:),coords(2,:),fx./fn)
title('Normalised?')

sgtitle(['width: ', num2str(radius),', depth: ', num2str(depth)])


%% Assigment lol
eps = 8.85*10^-12;
9.109*10^-31;
c = 3*10^8;
g = 10^5;
e = 1.602*10^-19;
me = 9.109*10^-31;
B = 1*10^-9;
6*pi*eps*me^3*c^3/(e^4*g*B^2) / (60*60*24*365)


%% 1D test

% bounds = 2;
% [x,y] = meshgrid(linspace(-bounds,bounds,50),linspace(-bounds,bounds,50));
% coords = [];
% for i = 1:size(x,1)^2
%     coords = [coords, [x(i);y(i)]];
% end
coords = [linspace(-2,2,100);zeros(1,100)];
location = [0;0];
radius = 1;
depth = 3;
z = depth*(((coords(1,:) - location(1)).^2 + (coords(2,:) - location(2)).^2)/radius^2 - 1);
zgrad = sum(2*depth/radius^2 * (coords - location),1).*(z<=0);
z = z.*(z<=0);
% plot3(coords(1,:),coords(2,:),sum(zgrad,1).*(z<=0),'.','markersize',5)
% title(['width: ', num2str(width),', depth: ', num2str(depth)])

% z = quadratic_rewards(coords,p);
figure
subplot(1,3,1)
plot3(coords(1,:),coords(2,:),z,'.')
title('Potential well')

subplot(1,3,2)
plot3(coords(1,:),coords(2,:),zgrad)
title('Gradient of potential (analytical)')

subplot(1,3,3)
plot3(coords(1,1:end-1),coords(2,1:end-1),diff(z)./diff(coords(1,:)))
title('Gradient of potential (numerical)')

sgtitle(['width: ', num2str(radius),', depth: ', num2str(depth)])

%% 2D truncated Gaussian well

[x,y] = meshgrid(linspace(-pi,pi,100),linspace(-pi,pi,100));
coords = [];
for i = 1:size(x,1)^2
    coords = [coords, [x(i);y(i)]];
end
location = [0,0];
sigma2 = 0.3;
radius = 1;
depth = 2;

distx = coords(1,:) - location(1);
disty = coords(2,:) - location(2);

% Potential function
z = depth * ((distx.^2 + disty.^2) / radius - 1);
truncated_z = -z .* (z <= 0);

% Gradient function
grad = -2 * depth * [distx, disty] / radius;

% Log gradient function


hold on
plot3(coords(1,:),coords(2,:),truncated_z)
title('Potential well')
legend('depth 1','depth 2')


%% Functions
function dist = norm1(vec)
    % Vec should be 2x10000 or smth
    dist = sqrt(vec(1,:).^2 + vec(2,:).^2);
end


