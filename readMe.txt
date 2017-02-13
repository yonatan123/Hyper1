
main:  perform estimation with support search, with and without acceleration. Outputs likelihood and total time of each method.  
withoutSupportSearch: perform estimation without support search.
_______________________________________________

RegularEM.m:EM algorithm without acceleration. 
AitkenEM.m: EM procedure with Aitken algorithm.
estimateParamVectors: perform estimation with 2 steps: support estimation, and EM of current support. 
modelLikelihood: log likelihood of observations y. 
searchSupport: estimate support of a, given observations and sparsity constant. 
