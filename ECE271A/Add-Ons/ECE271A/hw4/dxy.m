function d = dxy(x,y,sigma)
% compute the mahalanobis distance
d = (x - y) * inv(sigma) * transpose(x - y);
end
