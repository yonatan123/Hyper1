%Estimates the parameter vector a of noisy measurements Y=Xa+e,
% with Gaussian noise vector e. Y is  n x d observation matrix, X is n x p design marix, where the i-th column is 
%	normally distributed with mean F_i and covariance Q_i. The vector a has p non-negative entries that 
%  sum up to 1. Consists of 2 steps: support estimation, and EM of the
%  estimated support. 
%INPUT: y:  nxd observation matrix. 
%       F: nxp mean matrix, the means correspond to p normally distributed columns of the design matrix X. 
%		  varMatrix: nxnxp tensor, consists of p covariances of dimension nxn,
%							corresponding to the p normally distributed columns of X.
%		  sigma: variance of Gaussian noise vector e.  
%         x0: initial point for a
%         tol: tolerance
%         maxIter: maximum number of iterations
%         scheme: 'accelerate' for EM with Aitken acceleration, 'regular' -
%                  without acceleration. 
%OUTPUT: result-an estimate for a. 
function [result M1 M2]=estimateParamVector(y,F,varMatrix, sigma, sparsity,tolSearch,maxIterSearch,tol,maxIter,scheme); 
    
    [n p]=size(F);
    indexes=searchSupport(y,F,varMatrix, sigma, sparsity,tolSearch,maxIterSearch,scheme);   %find support of a
    
     
    indexes=sort(indexes); 

    %mean and covariance of current support
    newF=F(:,indexes);
    newvarMatrix=varMatrix(:,:,indexes);
    newp=length(indexes);
    varNoise=sigma*eye(n);
   
    
    if strcmp(scheme, 'accelerate')==1
        [vec M1 M2]=AitkenEM(y,newF,newvarMatrix,sigma,ones(newp,1)/newp,tol,maxIter);  %estimate a with current support
    end
    
    if strcmp(scheme, 'regular')==1
        [vec M1 M2]=RegularEM(y,newF,newvarMatrix,sigma,ones(newp,1)/newp,tol,maxIter);  %estimate a with current support
    end    
 
    result=zeros(p,1); 
    for i=1:newp
        result(indexes(i))=vec(i);
    end
    result;
end