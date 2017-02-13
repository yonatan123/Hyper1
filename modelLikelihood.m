%Compute likelihood of a, given the model  Y=Xa+e,
% with Gaussian noise vector e. Y is  n x d observation matrix, X is n x p design marix, where the i-th column is 
%	normally distributed with mean F_i and covariance Q_i. The vector a has p non-negative entries that 
%  sum up to 1. 
%INPUT: a: non-negative p x 1 parameter vector the sums up to 1.
%       y:  nxd observation matrix. 
%       F: nxp mean matrix, the means correspond to p normally distributed columns of the design matrix X. 
%		varMatrix: nxnxp tensor, consists of p covariances of dimension nxn,
%							corresponding to the p normally distributed columns of X.
%		sigma: variance of Gaussian noise vector e.  
%OUTPUT: f:  likelihood value. 
function [f]=modelLikelihood(a,y,F,varMatrix,sigma)

    p=size(varMatrix,3);
    n=size(varMatrix,1);
    
    %%compute covariance of y
    Qa=zeros(n);
    for i=1:p
        Qa=Qa+a(i)*a(i)*reshape(varMatrix(:,:,i),n,n);
    end
    Qa=Qa+sigma*eye(n);
    %%
    
    [d1 d2]=size(y);
    
    f=0;
    invQ=inv(Qa);
    detQ=det(Qa);
    
    if detQ==Inf
        detQ=10000;
    end
    for i=1:d2
      tempy=y(:,i);
      f=f+( .5*n*log(2*pi)+.5*log(  detQ  ) +.5*(tempy-F*a)'*invQ*(tempy-F*a)   );
    end  
end