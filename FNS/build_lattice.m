function locations = build_lattice(bounds)
    [x,y] = meshgrid(bounds,bounds);
    locations = [];
    for i = 1:size(x,1)*size(x,2)
        locations = [locations ; x(i),y(i)];
    end
end