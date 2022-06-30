function [screen_vec, trace] = bvGapSafeScreen(y, A, res, ATres, normA)
% Returns a n-dimensional vector taking values in {-1,0,+1} with:
% +1: coordinate surely saturated at upper limit.
% -1: coordinate surely saturated at lower limit.
%  0: undecisive.
%
% --Input arguments--
% Mandatory:
%   y: input vector (m x 1)
%   A: (m x n) matrix
% Recommended (for good performance):
%   res: residual (y-Ax in the LS case)
%   ATres: A.'*res
%   normA: (n x 1) vector containing the L2 norms of the columns of A

% Handle inputs (should be provided by the user for competitive results)
if nargin < 2, error('Two first input parameters (y and A) are mandatory.')
elseif nargin < 3, res = y - A*x;
elseif nargin < 4, ATres = A.'*res;
elseif nargin < 5, normA = sqrt(sum(A,1).^2).';
end


% Primal and dual cost functions (LS case)
primal = @(a) 0.5*sum( (y - a).^2 );
dual = @(b) 0.5*sum( y.^2 - (y - b).^2 );

% -- Dual update --
theta =  res; % simply the (generalized) residual
ATtheta = ATres;

% -- Duality gap --
gap = primal(y- res) - dual(theta); % gap has to be calculated anyway for GAP_Safe
gap(gap<=0) = eps;

% -- Screening --
radius = sqrt(2*gap);
screen_vec = sign(ATtheta).*(abs(ATtheta) - radius*normA > 0);

% Trace variables
trace.gap = gap;
% trace.nb_screen = sum(abs(screen_vec));
