function Pxy = CountProb2(x, mu, sigma, prior, c, di)

Pxy = 0;    
for i = 1 : c
    Pxy = Pxy + prior(i) * mvnpdf(x(:, 1:di), mu(i, 1:di), sigma{i}(1:di, 1:di));
end

end