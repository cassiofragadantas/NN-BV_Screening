function [V, output] = nnlsHALS_Screen(M,U,V,maxiter,options)

% Computes an approximate solution of the following nonnegative least 
% squares problem (NNLS)  
%
%           min_{V >= 0} ||M-UV||_F^2 
% 
% with an exact block-coordinate descent scheme [1] combined with 
% a safe screening strategy.  
%
% ****** Input ******
%   M  : m-by-N matrix 
%   U  : m-by-n matrix
%   V  : r-by-N initialization matrix 
%        default: one non-zero entry per column corresponding to the 
%        clostest column of U of the corresponding column of M 
%   maxiter: upper bound on the number of iterations (default=500).
%
%   *Remark. M, U and V are not required to be nonnegative. 
%
% ****** Output ******
%   V  : an r-by-N nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2
%
%
% [1] N. Gillis and F. Glineur, Accelerated Multiplicative Updates and 
% Hierarchical ALS Algorithms for Nonnegative Matrix Factorization, 
% Neural Computation 24 (4): 1085-1105, 2012.

if nargin <= 2 || isempty(V) 
    V = U\M; % Least Squares
    V = max(V,0); 
    alpha = sum(sum( (U'*M).*V ))./sum(sum( (U'*U).*(V*V'))); 
    V = alpha*V; 
end
if nargin <= 3, maxiter = 500; end
if nargin <= 4, options = []; end

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
    precalc.oracle_ATtheta = U.'*options.oracle_dual;
else
    precalc = [];
end


%[m, N] = size(M); 
[m,n] = size(U); 
UtU = U'*U; % n x n
UtM = U'*M; % n x N
UtUV = UtU*V;

% Trace output variables
output.time_it = zeros(1,maxiter);
output.nb_screen_it = zeros(1,maxiter);
if calc_gap, output.gap_it = zeros(1,maxiter); end

% Screening initialization
assert(isfield(options,'tdual'),'tdual should be provided');
tdual = options.tdual;
normU = sqrt(diag(UtU));
sumU = tdual.'*U;
screen_vec = false(size(V));


delta = 1e-15; % Stopping condition depending on evolution of the iterate V: 
              % Stop if ||V^{k}-V^{k+1}||_F <= delta * ||V^{0}-V^{1}||_F 
              % where V^{k} is the kth iterate. 
eps0 = 0; cnt = 1; eps = 1; 
startTime = tic;
while eps >= (delta)^2*eps0 && cnt <= maxiter %Maximum number of iterations
    nodelta = 0;
    
    %% Coordinate update loop    
    for k = 1 : n
        if ~screen_vec(k)
            deltaV = max((UtM(k,:)-UtUV(k,:))/UtU(k,k),-V(k,:)); %shouldn't it be V(k,:)/UtU(k,k) ?
            V(k,:) = V(k,:) + deltaV;

            UtUV = UtUV + UtU(:,k)*deltaV;

            nodelta = nodelta + deltaV*deltaV'; % used to compute norm(V0-V,'fro')^2;
            %if V(k,:) == 0, V(k,:) = 1e-16*max(V(:)); end % safety procedure
        end
    end
    
    if cnt == 1, eps0 = nodelta; end % improvement on first iteration
    eps = nodelta; % current improvement
    
    %% Safe screening
    if mod(cnt,screen_period) == 0
        % Notation: M - U*V
        [screen_new, ~] = nnGapSafeScreen(M, U, M - U*V, UtM - UtUV, normU, sumU,tdual,precalc); % could reuse input parameters
        if any(V(screen_new)~=0)
            UtUV = UtU(:,~screen_new)*V(~screen_new,:); 
        end
        V(screen_new,:) = 0;
        screen_vec = screen_vec | screen_new;
        output.nb_screen_it(cnt) = sum(screen_vec);
    end
    output.time_it(cnt) = toc(startTime);
    
    % Not executed normally! Compute gap for illustration-purpose only
    if calc_gap
        % Notation: M - U*V
        [~, trace] = nnGapSafeScreen(M, U, M - U*V, UtM - UtUV, normU, sumU,tdual,precalc); % could reuse input parameters
        output.gap_it(cnt) = trace.gap;
    end
    
    cnt = cnt + 1; 

end
output.time_it = output.time_it(1:cnt-1);
output.nb_screen_it = output.nb_screen_it(1:cnt-1);
if calc_gap, output.gap_it= output.gap_it(1:cnt-1); end

end % of function nnlsHALSupdt
