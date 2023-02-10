function [D,A]=kbdistance(X,S,scale,lambda)

%KBDISTANCE calculates the generalized Kullback-Leibler divergence
%indidually between each observation (row of X) and dictionary atom
%(column of S). Optionally it scales the dictionary atoms to that
%the distance is minimized.
%
%input:
%
%X           observation matrix (observations x features)
%S           basis matrix (atoms x features)
%scale       1=scale the dictionary atoms
%            0=do not scale them
%lambda      sparseness cost (observations x atoms, or a scalar)
%
%output 
%
%D           divergences (observations x atoms)
%A           scales (observations x atoms)
%            scale of each atom, when scaling has been used


%Tuomas Virtanen 2010

if any(X(:)<=0) || any(S(:)<=0)
  error('X and S need to be positive')
end


%distance=x*log(x/s)-x+s


%if s is scaled, the sum of x and s are the same
%-> the distance reduces to x*log(x/s*a) = xlog(x)-xlog(s)-xlog(a)


if ~scale
  if all(lambda==0)
    D=bsxfun(@plus,sum(X.*log(X)-X,2),sum(S,2)')-X*log(S)';
    A=ones(size(D));
  else
    error('not implemented yet for lambda>0')
  end
else
  A=bsxfun(@rdivide,sum(X,2),bsxfun(@plus,sum(S,2)',lambda));
  D=bsxfun(@minus,sum(X.*log(X),2),X*log(S)')-bsxfun(@times,sum(X,2),log(A))+sum(sum(bsxfun(@times,A,lambda)));
end

