%Demonstrate the algorithm with and without acceleration. 

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

%tolerance and max iterations for search and EM
tolSearch=10^-2;
maxIterSearch=100;
tol=10^-6; 
maxIter=1000; 
sparsity=3;


func=@(x)modelLikelihood(x,y,F,varMatrix,sigma); %likelihood w.r.t x, given y,F,varMatrix, sigma

%perform estimation without acceleration
scheme='regular';
tic
[result1 M1 M2]=estimateParamVector(y,F,varMatrix, sigma, sparsity,tolSearch,maxIterSearch,tol,maxIter,scheme);
time1=toc

%perform estimation with acceleration
scheme='accelerate';
tic
[result2 M1 M2]=estimateParamVector(y,F,varMatrix, sigma, sparsity,tolSearch,maxIterSearch,tol,maxIter,scheme);
time2=toc


display('without acceleration:')
display(strcat('time:    ', num2str(time1), '     logLikelihood:   ',num2str(func(result1))));

display('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
display('with acceleration:')
display(strcat('time:    ', num2str(time2), '     logLikelihood:   ',num2str(func(result2))));



