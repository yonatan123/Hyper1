%without search

%construct variables
n=200;
p=50;

F=randn(n,p);

 a=zeros(p,1);
 a(1:3)=1/3;
 %a(1)=0.45; a(2)=0.55;
 

 %covariance matrices
 varMatrix=zeros(n,n,p);
 punct=10^0;
 sigmavec=transpose( linspace(.2,.8,p) )/punct;
 for i=1:p
     %varMatrix(:,:,i)=sigmavec(i)*ones(n)+(1-sigmavec(i))*eye(n);
     %varMatrix(:,:,i)=sigmavec(i)*eye(n);
     temp=randn(n);
     varMatrix(:,:,i)=(temp'*temp);
 
 end
 sigma=.5;
 
 variance=sigma*eye(n);
for i=1:p
    variance=variance+a(i)*a(i)*reshape(varMatrix(:,:,i),n,n);
end

%construct observation vector y 
cases=1;
y=transpose(mvnrnd(F*a,variance,cases));

%tolerance and max iterations for EM
tol=10^-6; 
maxIter=1000; 


%EM algorithm 
tic 
[vec M1 M2]=AitkenEM(y,F,varMatrix,sigma,ones(p,1)/p,tol,maxIter);  %estimate a with current support
toc
