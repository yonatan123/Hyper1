%Find support (active set) of a in the model Y=Xa+e,
% with Gaussian noise vector e. Y is  n x d observation matrix, X is n x p design marix, where the i-th column is 
%	normally distributed with mean F_i and covariance Q_i. The vector a has p non-negative entries that 
%  sum up to 1, where sparsity s is assumes. 
%INPUT: y:  nxd observation matrix. 
%       F: nxp mean matrix, the means correspond to p normally distributed columns of the design matrix X. 
%		  varMatrix: nxnxp tensor, consists of p covariances of dimension nxn,
%							corresponding to the p normally distributed columns of X.
%		  sigma: variance of Gaussian noise vector e.  
%         sparsity: support size of a. 
%         tol: tolerance
%         maxIter: maximum number of iterations
%         scheme: 'accelerate' for EM with Aitken acceleration, 'regular' -
%                  without acceleration. 
%OUTPUT: indexes-  vector of the support indexes of a.
%                  length(indexes)=sparsity.
function indexes=searchSupport(y,F,varMatrix, sigma, sparsity,tol,maxIter,scheme)

    %initialization
    indexes=zeros(sparsity,1);
    [n p]=size(F);

    likelihoodVector=zeros(p,1); %each entry corresponds to the max likelihood after adding that index to the current support 
   
    tempEMvarMatrix=zeros(n,n,0);  %covariances of the current support 
    tempF=zeros(n,0);  %means of the current support
    tempEMF=zeros(n,0);
    varNoise=sigma*eye(n);
    func=@(x)modelLikelihood(x,y,F,varMatrix,sigma); %likelihood w.r.t x, given y,F,varMatrix, sigma
    
    for sp=1:sparsity
        
        %if current support is empty, check the likelihood of all 1-sparse vectors.
        if sp==1   
           for i=1:p
                temp=zeros(p,1);
                temp(i)=1;
                likelihoodVector(i)=func(temp);
            end
        end
   
         if sp>1
             for i=1:p
                 if length(find(i==indexes))>0
                     likelihoodVector(i)=Inf;  
                 else
                    tempvarMatrix=cat(3,tempEMvarMatrix,varMatrix(:,:,i)); %temporary covariance matrix of the current support and the current index
                    tempF=cat(2,tempEMF,F(:,i)); %temporary mean matrix of the current support and the current index
                    %
                    %compute estimate and its likelihood assuming current support
                    if strcmp(scheme,'accelerate')==1
                         [temp M1 M2]= AitkenEM(y,tempF,tempvarMatrix,varNoise,ones(sp,1)/sp,tol,maxIter); 
                    end
                    
                    if strcmp(scheme,'regular')==1
                         [temp M1 M2]= RegularEM(y,tempF,tempvarMatrix,varNoise,ones(sp,1)/sp,tol,maxIter); 
                    end
                    
                    
                    func=@(x)modelLikelihood(x,y,tempF,tempvarMatrix,sigma);
                    likelihoodVector(i)=func(temp); 
                 end 
             end
         end     
   
    %select index that yields maximal value
    [a1 b1]=min(likelihoodVector);
    indexes(sp)=b1(1);
    
    %update means and covariance of the new support
    tempEMvarMatrix=cat(3,tempEMvarMatrix,varMatrix(:,:,indexes(sp)));
    tempEMF=cat(2,tempEMF,F(:,indexes(sp)));
    
    end
end