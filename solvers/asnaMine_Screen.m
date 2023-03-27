function [A, output]=asnaMine_Screen(X,S,lambda,iterations,options)
%ASNA combined with safe screening. Modifications by Cassio F. Dantas 09/02/2023
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

if nargin < 5, options = []; end

if isfield(options,'calc_gap')
    calc_gap = options.calc_gap;
else
    calc_gap = false;
end
if isfield(options,'screen_period')
    screen_period = options.screen_period;
else
    screen_period = 10; %Screening tests are performed every screen_period iterations
end
if isfield(options,'oracle_dual')
    precalc.oracle_theta = options.oracle_dual;
    precalc.oracle_ATtheta = S.'*options.oracle_dual;
end

nObs = size(X,2);
nFeat = size(X,1);
nAtoms = size(S,2);


%normalize the scale of X
scale_X=sum(X,1);
X=bsxfun(@times,X,1./scale_X);


idx_X0 = false(size(X));
if any(X(:)==0)
  warning('The algorithm does not allow zero values of X. Any zero values are replaced with a small positive value')
  idx_X0 = (X==0);
  X(idx_X0)=1e-15;
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
    epsilon = 1e-6;
    gradA=kbdistanceRework(X,S+epsilon,lambda); % + epsilon to avoid errors on S=0
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

% Trace output variables
output.time_it = zeros(1,iterations);
output.nb_screen_it = zeros(1,iterations);
output.alpha_it = zeros(1,iterations);
if calc_gap, output.gap_it = zeros(1,iterations); end

% Screening initialization
assert(isfield(options,'tdual'),'tdual should be provided');
tdual = options.tdual;
normS = ones(size(S,2),1); % because normalisation was performed above. Otherwise, use scale_S
Stdual = tdual.'*S;
screen_vec = false(size(A));
precalc.min_y = min(X(~idx_X0));
precalc.sqrt_y = sqrt(X(~idx_X0));
denominator = min((1+sum(S(~idx_X0,:),1))./S(~idx_X0,:),[],2).^2;
precalc.alpha = min( X(~idx_X0)./denominator ); %coordinate-wise min
precalc.theta_old = zeros(size(X));
precalc.radius_old = inf;
precalc.idxy0 = idx_X0;


startTime = tic;

for i=2:iterations
  
  % Compute current approximation V + S_A W_A
  activeAtoms=find(any(A(:,updateframe),2));
  fullA=full(A(activeAtoms,updateframe));
  V(:,updateframe)=S(:,activeAtoms)*fullA + epsilon; % to avoid division by zero
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
        % zero-padding solution
        A_screen = A;
        A = zeros(nAtoms,nObs);
        A(~screen_vec) = A_screen;    

        %rescale back to the original scale of S and finish
        A=bsxfun(@times,A,double(scale_S'));
        A=bsxfun(@times,A,scale_X);

        % trim trace variables
        output.time_it(i) = toc(startTime);        
        output.time_it = output.time_it(1:i);
        output.nb_screen_it = output.nb_screen_it(1:i);
        output.alpha_it = output.alpha_it(1:i);
        if calc_gap, output.gap_it= output.gap_it(1:i); end
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

    %% Safe screening
    if mod(i-2,screen_period) == 0
        % Notation: X - S*A
        % R = X./V % But it is later modified
        % nnKLGapSafeScreen(y, A, res, ATres, normA, Atdual,t,Ax,precalc)
        [screen_new, precalc, ~] = nnKLGapSafeScreen(X, S, X./V - 1, -gradAll, normS, Stdual,tdual,V,precalc); % could reuse input parameters

        S(:,screen_new) = [];
        St(screen_new,:) = [];        
        Stdual(screen_new) = [];
        normS(screen_new) = [];
        A(screen_new,:) = [];
        OO(screen_new,:) = [];
        screen_vec(~screen_vec) = screen_new;

        output.nb_screen_it(i) = sum(screen_vec); 
        output.alpha_it(i) = precalc.alpha;
        if isfield(options,'oracle_dual'), precalc.oracle_ATtheta(screen_new) = []; end        
    end
    output.time_it(i) = toc(startTime);

    % Not executed normally! Compute gap for illustration-purpose only
    if calc_gap
        if mod(i,2)==0 % grad is recalculated only every 2 iterations
            [~,~,trace] = nnKLGapSafeScreen(X, S, X./V - 1, -gradAll, normS, Stdual,tdual,V,precalc); % could reuse input parameters
            output.gap_it(i) = trace.gap;
        else
            output.gap_it(i) = output.gap_it(i-1);
        end
    end
    

end


