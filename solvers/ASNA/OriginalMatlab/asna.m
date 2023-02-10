function A=asna(X,S,lambda,iterations)
%ASNA (Active-Set Newton Algorithm) minimizes the Kullback-Leibler
%divergence between non-negative observations X and the model A*S,
%plus L1-norm of A weighted by lambda, by estimating non-negative
%weights A.
%
%usage: A=asna(X,S,lambda,iterations)
%
%output 
%
%A          activations (observations x atoms)
%q          quality metrics
%
%input:
%
%X           non-negative observation matrix (observations x frequencies)
%S           non-negative dictionary matrix (atoms x frequencies)
%lambda      sparseness costs (scalar, or frames x atoms, or frames
%             x 1, or 1 x atoms)
%iterations  maximum number of iterations (default = 500)
%
%The algorithm is described in the following articles
%*T. Virtanen, J. F. Gemmeke, and B. Raj. Active-Set Newton
%Algorithm for Overcomplete Non-Negative Representations of Audio,
%IEEE Transactions on  Audio Speech and Language Processing,
%volume 21 issue 11, 2013.
%*T. Virtanen, B. Raj, J. F. Gemmeke, and H. Van hamme, Active-set
%Newton algorithm for non-negative sparse coding of audio, in proc.
%International Conference on Acoustics, Speech and Signal
%Processing, 2014, submitted for publication 
%
%See http://www.cs.tut.fi/~tuomasv/software.html for updated
%versions of the algorithm and updated information.

%Tuomas Virtanen 2012-2013

if (any(X(:)<0) || any(S(:)<0))
  error('X and S must be non-negative')
end


if ~exist('iterations','var')
  iterations=500;
end




%normalize the scale of X
scale_X=sum(X,2);
X=bsxfun(@times,X,1./scale_X);


if any(X(:)==0)
  warning('The algorithm does not allow zero values of X. Any zero values are replaced with a small positive value')
  X(X==0)=1e-15;
end


A=sparse(size(X,1),size(S,1));

%normalize the scale of S
scale_S=sqrt(sum(S.^2,2));
S=bsxfun(@times,S,1./scale_S);

%compensate by scaling lambda, so that the cost function is not
%affected by the above scaling
lambda=bsxfun(@times,lambda,1./scale_S');


%calculating SS in advance helps a lot
SS=S';
OO = bsxfun(@plus,ones(size(X))*SS,lambda);


%whether a frame is still updated or not
updateframe=1:size(A,1);


prevAadded=zeros(size(A,1),1);


for i=1:iterations
  
  if i==1
    %calculate a function which measures which entry of A alone
    %minimizes the divergence
    
    gradA=kbdistance(X,S,1,lambda);
    [tmp,maxi]=min(gradA,[],2);

    sumS=double(sum(S,2));
    if size(lambda,2)>1
      %pick the lambda values corresponding to the maximum gradient
      if size(lambda,1)>1
	lambdamaxi=lambda(sub2ind(size(lambda),(1:size(lambda,1))',maxi));
      else
	lambdamaxi=lambda(maxi)';
      end
      
    end
    
    A=sparse(1:size(A,1),maxi,bsxfun(@rdivide,sum(X,2),bsxfun(@plus,sumS(maxi),lambdamaxi)),size(A,1),size(A,2));

  elseif mod(i,2)==0 %every 2nd iteration add a new atom to the
                     %active set
    
    %add one element to A
    gradA=zeros(size(A));
    gradA(updateframe,:)=OO(updateframe,:)-R(updateframe,:)*SS;
    
    gradAall2=gradA;

    if mod(i,10)==0 %every 10th iteration check if a frame has
                    %converged and remove those from the calculations
      gradA(gradA>0)=0;
      gradnorm=sum(gradA.^2,2);
      updateframe(gradnorm(updateframe)<1e-15)=[];
      
      if isempty(updateframe) %if all the frames have converged
	%rescale back to the original scale of S
	A=bsxfun(@times,A,double(1./scale_S'));
	A=bsxfun(@times,A,scale_X);
	return
      end
    end

    
    %add only entries that are not yet active
    gradA(find(A(:)))=0; 
   
    [tmp,maxi]=min(gradA(updateframe,:),[],2);    

    ind=sub2ind(size(A),updateframe,maxi');

    %Do not add elements if the gradient is positive or very
    %small.
    ind(tmp>=-1e-10)=[];
    
    A(ind)=max(A(ind),1e-15);
    
  else
      
    clear gradAall2;
    
   end

  if i>1 %update A
  
    R2=X./V.^2;
    
    [z,j]=find(A);
    

    if ~exist('gradAall2','var')
      %Rcovariances=spprod(R,SS,z,j); % C version
      Rcovariances=sparseprod(R,SS,z,j);
    end

     
    for t=updateframe

      %aind=find(A(t,:));
      aind=j(z==t);
      

      if ~exist('gradAall2','var')
	gradA=OO(t,aind)-Rcovariances(z==t)';
      else
	gradA=gradAall2(t,aind);
      end


      %calculate the first gaussian inverse

      Sind=S(aind,:);
      GG=bsxfun(@times,R2(t,:),Sind)*Sind';
      
      
      %invGG=inv(GG+1e-10*eye(size(GG,2)));
      %newtonstep=gradA*invGG;
      newtonstep = ((GG+1e-10*eye(size(GG,2))) \ gradA')';


      zerograd=(newtonstep==0);
      maxstep=full(A(t,aind)./(newtonstep+zerograd));
      maxstep(newtonstep<=0)=Inf;
      stepsize=min(min(maxstep,[],2),1);
      stepsize(stepsize<0)=1;

      A(t,aind)=max(A(t,aind)-double(stepsize*newtonstep),0);
         
      
    end    


  end
  
  j2=find(any(A(updateframe,:),1));
  fullA=full(A(updateframe,j2));
  V(updateframe,:)=fullA*S(j2,:);


    
  R=X./V;

end

