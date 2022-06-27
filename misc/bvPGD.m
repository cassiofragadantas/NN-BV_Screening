function [x, output] = bvPGD(y,A,l,u,x,mu,maxiter,calc_gap,screen_period)
% Projected gradient algorithm for the Bounded-Variable Linear Regression
% problem. 
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
% - mu: step-size, if constant (2/norm(A)^2)
% - maxiter: maximum allowed number of iterations (1e6)
% - calc_gap: flag defining if the duality gap is to be computed (false)
% - screen_period: interval between screening tests (0, i.e. no screening)
%
% Author: Cassio F. Dantas
% Date: June 24th 2022

% Default input variables
if nargin < 4, error('bvPGD: Inputs {y, A, l, u} have to be provided'); end
if nargin < 5, x = zeros(size(A,2),1); end
if nargin < 6, mu=2/norm(A).^2; end
if nargin < 7, maxiter = 1e6; end
if nargin < 8, calc_gap = false; end
if nargin < 9, screen_period = 0; end

if screen_period % Safe screening activated
    calc_gap = true; % Gap has to be computed for screening
    normA = sqrt(sum(A.^2)).';
    saturated_coords = zeros(size(x)); % tracks the saturated coordinates
    n = length(x); u0 = u; l0=l;
end

step_strat = 'constant'; % 'constant' or 'Armijo' 

% Output variables
output.timeIt = zeros(1,maxiter);
output.costIt = zeros(1,maxiter);
if calc_gap, output.gapIt = zeros(1,maxiter); end
if screen_period, output.screenIt = zeros(1,maxiter); end

% Initializations
Ax = A*x;
startTime = tic;
k=1; 
gap = inf;
gap_tol = 1e-6;

% Cost functions (primal and dual)
% Euclidean distance
primal = @(yvec, a) 0.5*sum( (yvec - a).^2 );
dual = @(yvec,b,Atb,lvec,uvec) 0.5*sum( yvec.^2 - (yvec - b).^2 ) - lvec.'*min(0,Atb) - uvec.'*max(0,Atb);
% KL divergence (solver not implemented)
% epsAx = 0;
% primal = @(a) sum(v .* log(y ./ (a+epsAx)),'omitnan') + sum(- y + a + epsAx); % 0*log0 = 0
% dual = @(b, Atb) sum(y.*log(1+b),'omitnan') - sum(epsAx*b) - l.'*max(0,Atb) - u.'*min(0,Atb); % force 0*log(0) = 0 instead of NaN

if ~strcmp(step_strat,'constant') % Armijo strategy
    mu_tol = 1e-8*mu;
    f = primal(y,A*x);
end

% Main loop
while gap >= gap_tol && k <= maxiter
% delta_tol = 1e-8; delta_x = 1; delta_x0 = 0;    
% while delta_x >= delta_tol*delta_x0 && k <= maxiter
%     xprev = x;

    % -- Primal update --
    res = y - Ax;
    grad = -A.'*res;

    if strcmp(step_strat,'constant')
        x = x - mu*grad;
        x(x<l) = l(x<l); x(x>u) = u(x>u); % projection step (saturate entries)
        Ax = A*x;
    else
        %linesearch by armijo backtracking
        mu_k = 4*mu;
        xk = x - mu_k*grad;
        xk(xk<l) = l(xk<l); x(xk>u) = u(xk>u); % projection step (saturate entries)
        Axk = A*xk;
        fk = primal(y,Axk);

        while fk - f > 0.01*grad.'*(xk-x) && mu_k > mu_tol %0.5*mu_k
            % backstep mu_k
            mu_k = 0.5 * mu_k;

            % evaluate function at new iterate
            xk = x - mu_k*grad;
            xk(xk<l) = l(xk<l); x(xk>u) = u(xk>u); % projection step (saturate entries)
            Axk = A*xk;
            fk = primal(y,Axk);
        end
        if mu_k < mu_tol, keyboard; end
        f = fk;
%         mu = mu_k; % Much faster! To check
        x = xk;
        Ax= Axk;
    end
    
    if calc_gap
        % -- Dual update --
        theta =  res; % simply the (generalized) residual
        ATtheta = -grad;

        % -- Duality gap -- 
        if strcmp(step_strat,'constant'), f = primal(y,Ax); end
        output.costIt(k) =  f;
        gap = output.costIt(k) - dual(y,theta,ATtheta,l,u);
        gap(gap<=0) = eps;
        output.gapIt(k) = gap;    
    end
    
    % -- Stopping criterion --
%     delta_x = norm(x-xprev)^2;
%     if k == 1, delta_x0 = delta_x; end
    

    % Screening
    if mod(k,screen_period) == 0
        radius = sqrt(2*gap);
        screen_vec_l = (ATtheta + radius*normA < 0);
        screen_vec_u = (ATtheta - radius*normA > 0);

        y = y - A(:,screen_vec_l)*l(screen_vec_l) - A(:,screen_vec_u)*u(screen_vec_u);
        Ax = Ax - A(:,screen_vec_l)*x(screen_vec_l) - A(:,screen_vec_u)*x(screen_vec_u);    
        A(:,screen_vec_l | screen_vec_u) = [];    
        x(screen_vec_l | screen_vec_u) = []; 
        normA(screen_vec_l | screen_vec_u) = [];
        l(screen_vec_l | screen_vec_u) = [];
        u(screen_vec_l | screen_vec_u) = [];
        saturated_coords(~saturated_coords) = screen_vec_u - screen_vec_l;          
    end
    if screen_period, output.screenIt(k) = sum(abs(saturated_coords))/n; end

    output.timeIt(k) = toc(startTime);
    k = k+1;
end

output.costIt = output.costIt(1:k-1);
output.timeIt = output.timeIt(1:k-1);
output.gapIt = output.gapIt(1:k-1);
if screen_period, output.screenIt = output.screenIt(1:k-1); end

% zero-padding solution
if screen_period
    x_old = x;
    x = zeros(n,1);
    x(~saturated_coords) = x_old;
    x(saturated_coords == 1) = u0(saturated_coords == 1);
    x(saturated_coords == -1) = l0(saturated_coords == -1);
end
