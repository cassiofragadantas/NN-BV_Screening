function A=asnaRework(X,S,lambda,iterations)
%ASNA (Active-Set Newton Algorithm) minimizes the Kullback-Leibler
%divergence between non-negative observations X and the model S*A,
%plus L1-norm of A weighted by lambda, by estimating non-negative
%weights A.
%
%usage: A=asnaRework(X,S,lambda,iterations)
%
%output 
%
%A          activations (atoms x observations)
%
%input:
%
%X           non-negative observation matrix (frequencies x observations)
%S           non-negative dictionary matrix (frequencies x atoms)
%lambda      sparseness costs (scalar, or frames x atoms, or frames
%             x 1, or 1 x atoms)
%iterations  maximum number of iterations (default = 500)
%
%The algorithm on wich this implementation is based is described in the following articles
%*T. Virtanen, J. F. Gemmeke, and B. Raj. Active-Set Newton
%Algorithm for Overcomplete Non-Negative Representations of Audio,
%IEEE Transactions on  Audio Speech and Language Processing,
%volume 21 issue 11, 2013.
%*T. Virtanen, B. Raj, J. F. Gemmeke, and H. Van hamme, Active-set
%Newton algorithm for non-negative sparse coding of audio, in proc.
%International Conference on Acoustics, Speech and Signal
%Processing, 2014, submitted for publication 
%
%See https://P.SanJuan@gitlab.com/P.SanJuan/ASNA.git for latest version

%P. San Juan 2017 

if (any(X(:)<0) || any(S(:)<0))
  error('X and S must be non-negative')
end


if ~exist('iterations','var')
  iterations=500;
end

nObs = size(X,2);
nFeat = size(X,1);
nAtoms = size(S,2);


%normalize the scale of X
scale_X=sum(X,1);
X=bsxfun(@times,X,1./scale_X);


if any(X(:)==0)
  warning('The algorithm does not allow zero values of X. Any zero values are replaced with a small positive value')
  X(X==0)=1e-15;
end

%normalize the scale of S
scale_S=sqrt(sum(S.^2,1));
S=bsxfun(@times,S,1./scale_S);

%compensate by scaling lambda, so that the cost function is not
%affected by the above scaling
lambda=bsxfun(@times,lambda,1./scale_S');


%Precomputing some matrixes to speed up the gradient computations
St=S';
OO = bsxfun(@plus,St*ones(nFeat,nObs),lambda);


%wNon converged observations
updateframe=1:nObs;


 %calculate a function which measures which entry of A alone
    %minimizes the divergence
    
    gradA=kbdistanceRework(X,S,lambda); 
    [~,maxi]=min(gradA,[],1);

    sumS=double(sum(S,1));
    if size(lambda,1)>1
      %pick the lambda values corresponding to the maximum gradient
      if size(lambda,2)>1
	lambdamaxi=lambda(sub2ind(size(lambda),(1:size(lambda,2))',maxi));
      else
	lambdamaxi=lambda(maxi)';
      end
      
    end
    %Initialize active set 
    A=sparse(maxi,1:nObs,bsxfun(@rdivide,sum(X,1),bsxfun(@plus,sumS(maxi),lambdamaxi)),nAtoms,nObs);

for i=2:iterations
  
  % Compute current approximation V + S_A W_A
  activeAtoms=find(any(A(:,updateframe),2));
  fullA=full(A(activeAtoms,updateframe));
  V(:,updateframe)=S(:,activeAtoms)*fullA; 
  R=X./V;
  
  if mod(i,2)==0 %every 2nd iteration add a new atom to the
                     %active set for each nonconverged observation
    
    %Compute gradients w.r.t all weigths
    gradA=zeros(size(A));
    gradA(:,updateframe)=OO(:,updateframe)-St*R(:,updateframe);
    gradAll = gradA;
    atomNAded = false;
    
    if mod(i,10)==0 %every 10th iteration check if a frame has
                    %converged and remove those from the calculations
      gradA(gradA>0)=0;
      gradnorm=sum(gradA.^2,1);

      updateframe(gradnorm(updateframe)<1e-15)=[];
  

      if isempty(updateframe) %if all the frames have converged
        %rescale back to the original scale of S and finish
        A=bsxfun(@times,A,double(scale_S'));
        A=bsxfun(@times,A,scale_X);
        return
      end
    end

    
    %add only entries that are not yet active
    gradA(A~=0)=0; 
   
    [tmp,maxi]=min(gradA(:,updateframe));    
    
    ind=sub2ind(size(A),maxi,updateframe);

    %Do not add elements if the gradient is positive or very
    %small.
    ind(tmp>=-1e-10)=[];
    
    A(ind)=max(A(ind),1e-15);
    
  else
      
    atomNAded= true;
    
  end
  
   
    
    [sparseRows,sparseCols]=find(A);

    %if all gradient have not been computed previously in this iteration compute the sparse product
    if atomNAded 
      StR=sparseprodRework(S,R,sparseRows,sparseCols); 
    end
    
    R=R./V;
    for t=updateframe

      aind=sparseRows(sparseCols==t);
      
      %Pick the active gradients for t if already computed, compute them otherwise  
      if atomNAded
        gradA=OO(aind,t)-StR(sparseCols==t);
      else
        gradA=gradAll(aind,t);
      end


      %Compute the hessian
      Sind=S(:,aind);
      Hwa= Sind' * bsxfun(@times,R(:,t),Sind);
      
      %Solve the sistem of equations Hwa x newtonstep = ghradA to obtain
      %the step direction
      C = chol(Hwa +1e-10*eye(size(Hwa,2)));
      y = C'\gradA;
      newtonstep = (C\y);
      
      %Compute step size
      maxstep=full(A(aind,t)./newtonstep);
      maxstep(newtonstep<=0)=Inf;
      stepsize=min(min(maxstep),1);
      stepsize(stepsize<0)=1;

      %update weigths for active atoms of t, remove negative or 0
      A(aind,t)=max(A(aind,t)-double(stepsize*newtonstep),0);

    end    


end


