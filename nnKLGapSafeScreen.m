function [screen_vec, precalc, trace] = nnKLGapSafeScreen(y, A, res, ATres, normA, Atdual,t,Ax,precalc)
% Returns a n-dimensional logic vector with true entries indicating the
% saturated (zero) coordinates.
%
% --Input arguments--
% Mandatory:
%   y: input vector (m x 1)
%   A: (m x n) matrix
% Recommended (for good performance):
%   res: -df(Ax)/d(Ax) (y-Ax in the LS case, (y-Ax)/Ax in the KL case)
%   ATres: A.'*res = -df/dx
%   normA: (n x 1) vector containing the L2 norms of the columns of A
%   Atdual: (n x 1) vector containing the dot product A.'*tdual
%   t: (n x 1) dual direction vector tdual
%   Ax: (n x 1) vector product A*x

% Handle inputs (should be provided by the user for competitive results)
if nargin < 2, error('Two first input parameters (y and A) are mandatory.')
elseif nargin < 3, res = y/Ax -1; %KL case. y - A*x; %LS case
elseif nargin < 4, ATres = A.'*res;
elseif nargin < 5, normA = sqrt(sum(A,1).^2).';
elseif nargin < 6, Atdual = sum(A,1); 
elseif nargin < 7, t = 1; 
elseif nargin < 8, Ax = y./(res+1); %KL case. Ax = y-res; %LS case
end

% Primal and dual cost functions 
primal = @(a) sum(y.*log(y./a) - y + a); % KL distance (fixing first variable as y)
% primal = @(a) sum(y.*log(y./(a + param.epsilon)) + - y + a + param.epsilon); % with param.epsilon
% primal = @(a) sum(y(y~=0).*log(y(y~=0)./a(y~=0))) + sum(- y + a); % force 0*log(0) = 0 (instead of NaN) 
dual = @(b) y.'*log(1+b); % - sum(param.epsilon*b);
% dual = @(b) y(y~=0).'*log(1+b(y~=0)) - sum(param.epsilon*b); % Avoids 0*log(0) = NaN

% -- Dual update --
% Dual feasible point
% 1) Dual translation (to fall inside dual feasible set)
epsilon = max(ATres./Atdual.'); %max(ATres) also works if A is normalized, but is slightly worse.
theta = res - epsilon*t;
% 2) Rescale (to fall inside dual function domain)
theta = theta/max(-min(theta)+1e-6,1);

ATtheta = ATres - epsilon*Atdual.'; %= A.'* theta; Should be zero at coordinates xj ~= 0 and negative otherwise

% -- Duality gap --
gap = primal(Ax) - dual(theta); % gap has to be calculated anyway for GAP_Safe
gap(gap<=0) = eps;

if exist('precalc','var')
    % -- Strong concavity bound --    
    % Project theta into previous safe sphere
    theta_dist = norm(theta - precalc.theta_old); 
    if (theta_dist > precalc.radius_old)
        theta = precalc.theta_old + precalc.radius_old*(theta-precalc.theta_old)/theta_dist;
    end
    % Compute fixed-point
    % improv_flag = ( theta_dist > precalc.radius_old*(sqrt(gap/gap_last_alpha)-1) ); % tests if improvement in alpha is possible
    if gap < precalc.min_y/2 % && improv_flag 
        denominator = (1 + theta(~precalc.idxy0)).^2 ;
        alpha_star = min((precalc.sqrt_y-sqrt(2*gap)).^2./denominator);
        precalc.alpha = max(alpha_star, precalc.alpha);    
    end

else
    precalc.alpha = eps; % very low, for security. Bad performance.
end

% -- Screening --
radius = sqrt(2*gap/precalc.alpha);
screen_vec = (ATtheta + radius*normA < 0);

precalc.theta_old = theta;
precalc.radius_old = radius;

% Trace variables
trace.gap = gap;