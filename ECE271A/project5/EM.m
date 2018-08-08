function [mu_c,sigma_c,pi_c,iter]=EM(D,C)
%Expectation-Maximization Algorithm
%D is training set, each row is an example
%C is # of components
n=size(D,1);%number of examples
dim=size(D,2);%dimension of each example
%Initialize pi_c
pi=zeros(1,C);%1*C matrix
pi=pi+1/C;% assume pi~U, can be others
%Initialize mu_c
r=randi([1 n],1,C);%#C random number between[1~n]
mu=D(r(:),:);%C*dim matrix, each line is a class mean
%Initialize sigma_c
sigma = zeros(dim,dim,C);
for i =1:C
    alpha = 20+10*rand(1,dim);
    I = eye(dim);
    sigma(:,:,i) = alpha.*I;
end   



%h[i,j]:P(class=j | X_i)
Jt=zeros(n,C);%Joint Distribution P(X_k,Class)
for iter=1:1000
    %E-step
    for i=1:C
        %class j, k_th example
        %G(X_k;mu_j,sigma_j)*pi_j:P(X_k|Class=j)*P(Class=j)
        Jt(:,i)=mvnpdf(D,mu(i,:),sigma(:,:,i))*pi(i);    
    end
    %P(Class=j|X_k,theta)=P(X_k,Class=j)/P(X_k)
    %h_kj={G(X_k;mu_j,sigma_j)*pi_j}/SUM(all class_j{G(X_k;mu_j,sigma_j)*pi_j})
    %log{P(Data|theta)}:log-likelihood
    lkhd(iter)=sum(log(sum(Jt,2)));
    %h(i,j)=P(class=j|X_i);
    h=Jt./sum(Jt,2);%sum of colums, column vextor
    
    %M-step
    %new pi
    pi=sum(h)/n;
    %n_hat_j=sum(h_k_j for all k=1~n)
    %mu_j=(1/n_hat_j)*sum{h_k_j*X_k for all k=1~n}
    %mu=H'*X./sum(H)
    mu = (h'*D)./sum(h)';
    for class_j = 1:C
        %sigma_j=(1/n_hat_j)*sum{h_k_j*(X_k-mu_j)^2 for all k=1~n}
        sigma_sub = (D-mu(class_j,:))'.*h(:,class_j)'*(D-mu(class_j,:));
        sigma(:,:,class_j)  = diag(diag((sigma_sub./sum(h(:,class_j),1))+0.0000001));
    end
    
    if iter<2
       continue;
    end
    if abs(lkhd(iter) - lkhd(iter-1))<0.001
       break; 
    end
end
mu_c=mu;
sigma_c=sigma;
pi_c=pi;





