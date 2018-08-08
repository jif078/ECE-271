function d = dxy(x,y,sigma)

d = (x - y) * inv(sigma) * transpose(x - y);
end
