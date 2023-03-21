function [screen_vec, trace] = nnGapSafeScreen(y, A, res, ATres, normA, sumA,t,precalc)
% Returns a n-dimensional logic vector with true entries indicating the
% saturated (zero) coordinates.
%
% --Input arguments--
% Mandatory:
%   y: input vector (m x 1)
%   A: (m x n) matrix
% Recommended (for good performance):
%   res: residual (y-Ax in the LS case)
%   ATres: A.'*res
%   normA: (n x 1) vector containing the L2 norms of the columns of A
%   sumA: (n x 1) vector containing the columwise sum of the entries of A 

% Handle inputs (should be provided by the user for competitive results)
if nargin < 2, error('Two first input parameters (y and A) are mandatory.')
elseif nargin < 3, res = y - A*x;
elseif nargin < 4, ATres = A.'*res;
elseif nargin < 5, normA = sqrt(sum(A,1).^2).';
elseif nargin < 6, sumA = sum(A,1); 
elseif nargin < 7, t = 1; 
end

% Primal and dual cost functions (LS case)
primal = @(a) 0.5*sum( (y - a).^2 );
dual = @(b) 0.5*sum( y.^2 - (y - b).^2 );

% -- Dual update --
if exist('precalc','var') && isfield(precalc,'oracle_theta')
    % Oracle dual point
    theta = precalc.oracle_theta;
    ATtheta = precalc.oracle_ATtheta; 
else
    % Dual feasible point    
    epsilon = max(ATres./sumA.'); %max(ATres) also works if A is normalized, but is slightly worse.
    theta = res - epsilon*t;
    ATtheta = ATres - epsilon*sumA.'; %= A.'* theta; Should be zero at coordinates xj ~= 0 and negative otherwise    
end


% -- Duality gap --
gap = primal(y-res) - dual(theta); % gap has to be calculated anyway for GAP_Safe
gap(gap<=0) = eps;

% -- Screening --
radius = sqrt(2*gap);
screen_vec = (ATtheta + radius*normA < 0);

% Trace variables
trace.gap = gap;