function [x, output] = ChamPockPGD(y,A,l,u,x,maxiter,L,calc_gap,screen_period,options)
% Primal-dual Chambolle-Pock algorithm for the Bounded-Variable Linear 
% Regression problem. 
%
% min_{x | l_i <= x_i <= u_i} D(y, Ax)
%
% where the cost function D(y,Ax) is a beta-divergence between vectors y
% and Ax (for now, only the case beta=2, i.e. the LS-case is implemented)
%
% Required inputs:
% - y: (m x 1) input vector (to be approximated by the linear model A*x)
% - A: (m x n) matrix
% - l, u: (n x 1) vectors with the lower and upper bounds (respectively)
%         for the solution vector x.
% Optional inputs (default values):
% - x: (n x 1) initialization vector (all-zeros vector )
% - maxiter: maximum allowed number of iterations (1e6)
% - L: operator norm of matrix A  (norm(A))
% - calc_gap: flag defining if the duality gap is to be computed (false)
% - screen_period: interval between screening tests (0, i.e. no screening)
%
% Solver parameters:
% - tau, sigma: related to primal and dual updates respectively. Should 
%               satisfy sigma*tau*norm(A)^2<1
% - alpha=1 for the ADMM, but can be set in [0,1].
%
% Author: Cassio F. Dantas
% Date: Mars 24th 2023
%
% Partially based on the implementation by Gabriel Peyre available at
% http://www.numerical-tours.com/
%
% Uses the Preconditioned Alternating direction method of multiplier (ADMM) 
% method described in 
%  [1]  A. Chambolle, T. Pock, "A first-order primal-dual algorithm for 
%       convex problems with applications to imaging, Preprint CMAP-685	


% Default input variables
if nargin < 4, error('ChamPockPGD: Inputs {y, A, l, u} are mandatory'); end
if nargin < 5, x = zeros(size(A,2),1); end
if nargin < 6, maxiter = 1e6; end
if nargin < 7, L = norm(A); end
if nargin < 8, calc_gap = false; end
if nargin < 9, screen_period = 0; end
if nargin < 10, options = []; end
if isfield(options,'oracle_dual')
    options.oracle_ATtheta = A.'*options.oracle_dual;
end


if calc_gap
    stop_crit = 'gap';
else
    stop_crit = 'delta'; % 'gap' for duality gap criterion
end

if screen_period % Safe screening activated: initializations
    calc_gap = true; % Gap has to be computed for screening
    normA = sqrt(sum(A.^2)).';
    saturated_coords = zeros(size(x)); % tracks the saturated coordinates
    n = length(x); u0 = u; l0=l;
end

if strcmp(stop_crit,'gap'), calc_gap = true; end

% Output variables
output.timeIt = zeros(1,maxiter);
output.costIt = zeros(1,maxiter);
if calc_gap, output.gapIt = zeros(1,maxiter); end
if screen_period, output.screenIt = zeros(1,maxiter); end

% Initializations
Ax = A*x;
z = Ax-y; z1 = Ax-y;
ATz1 = A.'*Ax; %Ax1 = Ax;
startTime = tic;
k=1; 
gap = inf;
converged = false;  % Stopping criterion:
gap_tol = 1e-6;     % used if (stop_crit == 'gap')
delta_tol = 1e-19;  % used otherwise

% Solver parameters
alpha = 1;
% sigma = 1/L; tau = .9/L;
sigma = .01; tau = .9/(sigma*L^2);
% tau = 1; sigma = .9/(tau*L^2);
gam = 1; % For acceleration. F^* is 1/gam grad-Lipschitz

% Cost functions (primal and dual)
% Euclidean distance
primal = @(yvec, a) 0.5*sum( (yvec - a).^2 );
dual = @(yvec,b,Atb,lvec,uvec) 0.5*sum( yvec.^2 - (yvec - b).^2 ) - lvec.'*min(0,Atb) - uvec.'*max(0,Atb);
% KL divergence (solver not implemented)
% epsAx = 0;
% primal = @(a) sum(v .* log(y ./ (a+epsAx)),'omitnan') + sum(- y + a + epsAx); % 0*log0 = 0
% dual = @(b, Atb) sum(y.*log(1+b),'omitnan') - sum(epsAx*b) - l.'*max(0,Atb) - u.'*min(0,Atb); % force 0*log(0) = 0 instead of NaN

% Main loop
while ~converged && k <= maxiter

    xold = x; % previous iterate
    zold = z;

    % -- Primal-dual update --
    x = x-tau*ATz1; x(x<l) = l(x<l); x(x>u) = u(x>u); % x = ProxG(x-tau*ATz1, tau)
    Ax = A*x;
    z = (z+sigma*(Ax - y))/(1+sigma); % z = ProxFS(z+sigma*Ax, sigma);
%     % Acceleration (not working properly. Usually slower)
%     alpha=1/sqrt(1+2*gam*tau);
%     tau=tau/alpha;      % *alpha
%     sigma=sigma*alpha;  % /alpha
    z1 = z + alpha * (z-zold);
    ATz1 = A.'*z1;

    % Reverse order (extrapolation on the primal), no acceleration
%     z = (z+sigma*(Ax1 - y))/(1+sigma); % ProxFS( z+sigma*K(x1), sigma);
%     x = x-tau*ATz; x(x<l) = l(x<l); x(x>u) = u(x>u); % ProxG(  x-tau*KS(z), tau);
%     x1 = x + alpha * (x-xold);
    
    % -- Stopping criterion --
    if calc_gap
        % -- Dual update --
        if isfield(options,'oracle_dual')
            theta = options.oracle_dual;
            ATtheta = options.oracle_ATtheta;
        else
            theta =  -z1; % simply the (generalized) residual
            ATtheta = -ATz1;
        end

        % -- Duality gap -- 
        output.costIt(k) =  primal(y,Ax);
        gap = output.costIt(k) - dual(y,theta,ATtheta,l,u);
        gap(gap<=0) = eps;
        output.gapIt(k) = gap;    
    end
    
    if strcmp(stop_crit,'gap') % duality gap
        converged = (gap < gap_tol);
    else % variation on the solution estimate
        delta_x = norm(x-xold)^2;
        if k == 1, delta_x0 = delta_x; end
        converged = (delta_x < delta_tol*delta_x0);
    end

    % -- Screening --
    if mod(k,screen_period) == 0
        radius = sqrt(2*gap);
        screen_vec_l = (ATtheta + radius*normA < 0);
        screen_vec_u = (ATtheta - radius*normA > 0);

        y = y - A(:,screen_vec_l)*l(screen_vec_l) - A(:,screen_vec_u)*u(screen_vec_u);
%         Ax = Ax - A(:,screen_vec_l)*x(screen_vec_l) - A(:,screen_vec_u)*x(screen_vec_u);    
        A(:,screen_vec_l | screen_vec_u) = [];    
        x(screen_vec_l | screen_vec_u) = []; 
        normA(screen_vec_l | screen_vec_u) = [];
        l(screen_vec_l | screen_vec_u) = [];
        u(screen_vec_l | screen_vec_u) = [];
        saturated_coords(~saturated_coords) = screen_vec_u - screen_vec_l;          
        if isfield(options,'oracle_dual'), options.oracle_ATtheta(screen_vec_l | screen_vec_u) = []; end
        ATz1(screen_vec_l | screen_vec_u) = [];
    end
    if screen_period, output.screenIt(k) = sum(abs(saturated_coords))/n; end

    output.timeIt(k) = toc(startTime);
    k = k+1;
end

output.costIt = output.costIt(1:k-1);
output.timeIt = output.timeIt(1:k-1);
if calc_gap, output.gapIt = output.gapIt(1:k-1); end
if screen_period, output.screenIt = output.screenIt(1:k-1); end

% zero-padding solution
if screen_period
    xold = x;
    x = zeros(n,1);
    x(~saturated_coords) = xold;
    x(saturated_coords == 1) = u0(saturated_coords == 1);
    x(saturated_coords == -1) = l0(saturated_coords == -1);
end
