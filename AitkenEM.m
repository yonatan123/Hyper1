%EM algorithm with Aitken acceleration. Estimates the parameter vector a of noisy measurements Y=Xa+e,
% with Gaussian noise vector e. Y is  n x d observation matrix, X is n x p design marix, where the i-th column is 
%	normally distributed with mean F_i and covariance Q_i. The vector a has p non-negative entries that 
%  sum up to 1. 
%INPUT: y:  nxd observation matrix. 
%       F: nxp mean matrix, the means correspond to p normally distributed columns of the design matrix X. 
%		  varMatrix: nxnxp tensor, consists of p covariances of dimension nxn,
%							corresponding to the p normally distributed columns of X.
%		  sigma: variance of Gaussian noise vector e.  
%         x0: initial point for a
%         tol: tolerance
%         maxIter: maximum number of iterations
%OUTPUT: result-an estimate for a. 
function [result EA EAtA]=AitkenEM(y,F,varMatrix,sigma,x0,tol,maxIter)


    obs=size(y,2);
    [n p]=size(F);
    
    
    %rearrange input matrices F,y and varMatrix for further use
    FF=kron(ones(obs,1),F); 
    yy=reshape(y,size(y,1)*size(y,2),1); 
    UU=transpose(reshape(permute(varMatrix,[2 1 3]),n,[]));
       
    %initialization of latent conditional expectation matrices
    EA=zeros(obs*n,p);  
    EAtA=zeros(p);      
    prev=Inf*ones(p,1); %previous result initialization
	count=0;
    
    k=3;
    iterationMatrix=zeros(p ,k);  %iterationMatrix- 3 last iterations result to be used in  Aitken updating 
    countk=1;  %iteration counting
	
    
  
    while norm(prev-x0)>tol  & count<maxIter 
        countk;
        
        %%
        %Aitken updating after 3 consecutive iterations
        if countk==k+1
           %Aitken updating
           if nnz((iterationMatrix(:,3)-2*iterationMatrix(:,2)+iterationMatrix(:,1)))==p
                x0=iterationMatrix(:,1)-((iterationMatrix(:,2)-iterationMatrix(:,1)).^2)./(iterationMatrix(:,3)-2*iterationMatrix(:,2)+iterationMatrix(:,1));
           end
           countk=1;
        end
        %%
        %% EM Updating
        if countk<=k
                
           prev=x0;  %update previous result
           count=count+1;  %update iteration number
           
           %calculate inverse covariance of y, given varMatrix and sigma
           tempVariance=zeros(n);  
           z=reshape(repmat(x0',n*n,1),n,n,p);
           tempVariance=sum(z.*z.*varMatrix,3)+sigma*eye(n);
           tempVariance=inv(tempVariance);

           %%%%E-step:calculate conditional expectations
           %EA-> E(X|y)
           %EAtA-> E(X'X|y)
           
           tempr=y-diag(F*x0)*ones(size(y));
           for i=1:p
               EA(:,i)=FF(:,i)+x0(i)*reshape((varMatrix(:,:,i)*tempVariance)*(tempr),n*obs,1); 
           end
           EAtA=transpose(EA)*EA;  
           traceMat=zeros(p);
           tempA=UU*tempVariance;
                
           for i=1:p
                start=1+(i-1)*n; stop=i*n;
                tempmat=tempA(start:stop,:);
                %
                for j=1:p
                    traceMat(i,j)=sum(sum(  tempmat.*  varMatrix(:,:,j)   ));
                    traceMat(j,i)=traceMat(i,j);
                end
                %
           end
           
           EAtA=EAtA-(x0*x0').*traceMat*obs;
           for i=1:p
               EAtA(i,i)=EAtA(i,i)+trace(obs* varMatrix(:,:,i));
           end

           
           %M-step: minimize quadratic program
           options =  optimset('Display','none');
           x0=quadprog(EAtA,-yy'*EA,[],[],ones(1,p),[1],zeros(p,1));
                     
                  
   
          iterationMatrix(:,countk)= x0; %update result
          countk=countk+1;
        end
    
    end

    result=x0;

end