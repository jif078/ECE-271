function pb=EM_probability(X,mu,sigma,pi)
pb=0;
for i=1:size(mu,1)
    pb=pb+mvnpdf(X,mu(i,:),sigma(:,:,i))*pi(i);
end