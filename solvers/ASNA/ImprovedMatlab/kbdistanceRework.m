function [D,A]=kbdistanceRework(X,S,lambda)

%KBDISTANCE calculates the generalized Kullback-Leibler divergence
%indidually between each observation (col of X) and dictionary atom
%(row of S). Optionally it scales the dictionary atoms to that
%the distance is minimized.
%
%input:
%
%X           observation matrix (features x observations)
%S           basis matrix (features x atoms)
%lambda      sparseness cost (observations x atoms, or a scalar)
%
%output 
%
%D           divergences (atoms x observations)
%A           minimum atoms (atoms x observations)
%            value of each atom ehich minimizes the divergence


%P. San Juan 2017

if any(X(:)<=0) || any(S(:)<=0)
  error('X and S need to be positive')
end


%distance=x*log(x/s)-x+s


%if s is scaled, the sum of x and s are the same
%-> the distance reduces to x*log(x/s*a) = xlog(x)-xlog(s)-xlog(a)


  A=bsxfun(@rdivide,sum(X,1),bsxfun(@plus,sum(S,1)',lambda));
  D=bsxfun(@minus,sum(X.*log(X),1),log(S)'*X)-bsxfun(@times,sum(X,1),log(A))+sum(sum(bsxfun(@times,A,lambda)));


