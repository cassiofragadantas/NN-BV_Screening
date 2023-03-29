function [x, obj, x_it, screen_it, stop_crit_it, trace] = CoD_KL_l1_GAPSafe(A, y, lambda, x0, param, precalc)
% KL_l1_MM a Majoration-minimization approach to solve a non-negative 
% l1-regularized problem that uses the Kullback-Leibler divergence as the 
% data-fidelity term :
% 
% (1)                  min_(x>=0) D_KL(y, Ax) + lambda*norm(x,1)
%
% where lambda is a sparsity regularization term D_KL(a,b) is the Kullback 
% Leibler divergence between vectors a and b, which, in turn, can be 
% written in terms of a scalar divergence between each of the entries a_i, 
% b_i of vectors a and b, as follows:
% 
%   d_KL(a_i,b_i) = a_i log(a_i/b_i) - a_i + b_i
%
% This functions therefore seeks a sparse vector x such that y ≈ Ax in the
% Kullback-Leibler sense.
%
%   Required inputs:
%       A   : (n x m) matrix.
%       y   : (n x 1) input vector.
%       lambda   : regularization parameter (see (1)).
%
%   Optional inputs:
%       x0    : Initialization of the vector x of size (m x 1).
%               (Default: ones(m,1))
%       param : Matlab structure with some additional parameters.
%           param.verbose   : Verbose mode (Default: false)
%           param.TOL       : Stop criterium. When ||x(n) - x(n-1)||_2, 
%                             where n is the iteration number, is less than
%                             TOL, we quit the iterative process. 
%                             (Default: 1e-10).
%           param.MAX_ITER  : Stop criterium. When the number of iterations
%                             becomes greater than MAX_ITER, we quit the 
%                             iterative process. 
%                             (Default: 1000).
%           param.constraint: A function handle imposing some constraint on
%                             x. 
%                             (Default: @(x) x).
%           param.save_all  : Flag enabling to return the solution estimate
%                             at every iteration (in the variable x_it)
%                             and the objective function (in variable obj)
%                             (Default: false).
%         
%   Output:
%       x     : (m x 1) vector, solution of (1)
%
%   Optional outputs (only when param.save_all == true)
%       obj  : A vector with the objective function value at each iteration
%       x_it : matrix containing in its columns the solution estimate at
%              each iteration.
%       screen_it : binary matrix containing as its columns the screened 
%                   coordinates ate each iteration.
%
%   See also:
%
%   References:
%
% Author: Cassio F. Dantas
% Date: 17 Mar 2020

%% Input parsing

assert(nargin >= 2, 'A (matrix), y (vector) and lambda (scalar) must be provided.');
% x0 = A.'*y; % If x0 not provided
A = full(A); y = full(y);%current C implementation does not support sparse matrices

% problem dimensions (n,m)
assert(length(size(A)) == 2, 'Input variable A should be a matrix')
[n, m] = size(A);
assert(all(size(y) == [n 1]),'Input variable y must be a (n x 1) vector')
assert(all(y >= 0),'Input variable y must be non-negative')
assert(isscalar(lambda) & lambda >= 0,'Input variable lambda must non-negative scalar')

% x0
if (nargin < 4) || isempty(x0); x0 = ones(m,1); end
assert(all(size(x0) == [m 1]),'x0 must be a (m x 1) vector')

% param
if (nargin < 5); param = []; end
if ~isfield(param, 'verbose'); param.verbose = false; end
if ~isfield(param, 'TOL'); param.TOL = 1e-10; end
if ~isfield(param, 'MAX_ITER'); param.MAX_ITER = 10000; end
if ~isfield(param, 'save_all'); param.save_all = false; end
if ~isfield(param, 'save_time'); param.save_time = true; end
if ~isfield(param, 'stop_crit'); param.stop_crit = 'difference'; end
if ~isfield(param, 'epsilon'); param.epsilon = 0; end
if ~isfield(param, 'screen_period'); param.screen_period = 10; end
if ~isfield(param, 'calc_gap'); param.calc_gap = false; end %TODO
if isfield(param,'oracle_dual')
    precalc.oracle_theta = param.oracle_dual;
    precalc.oracle_ATtheta = A.'*param.oracle_dual;
end

if strcmp(param.stop_crit, 'gap'); param.calc_gap = true; end

% nb_portions = 10;
% idx_portions = round(linspace(0,m,nb_portions+1));

% objective function
%f.eval = @(a,b) sum(a.*log(a./b) - a + b); % KL distance
f.eval = @(a) sum(y(y~=0).*log(y(y~=0)./(a(y~=0)+ param.epsilon))) + sum(- y + a + param.epsilon); % force 0*log(0) = 0 (instead of NaN) 
g.eval = @(a) lambda*norm(a, 1); % regularization

tStart = tic;
%% Initialization
k = 0;  % Iteration number
x = x0 + 0; % Signal to find (+0 avoid x to be just a pointer to x0)
x_old = inf;
stop_crit = Inf; % Difference between solutions in successive iterations
% screen_vec = false(size(x));
rejected_coords = false(m,1);

idx_y0 = (y==0);

Ax = A*x; % For first iteration
radius = inf; theta = 0;

% Screening initialization
assert(isfield(param,'tdual'),'tdual should be provided');
tdual = param.tdual;
normA = sqrt(sum(A.^2,1)).';
Atdual = tdual.'*A;
precalc.min_y = min(y(~idx_y0));
precalc.sqrt_y = sqrt(y(~idx_y0));
denominator = min((1+sum(A(~idx_y0,:),1))./A(~idx_y0,:),[],2).^2;
precalc.alpha = min( A(~idx_y0)./denominator ); %coordinate-wise min
precalc.theta_old = zeros(size(y));
precalc.radius_old = inf;
precalc.idxy0 = idx_y0;

if param.save_all
    obj = zeros(1, param.MAX_ITER); % Objective function value by iteration
    obj(1) =  f.eval(Ax) + g.eval(x) ;
    screen_it = false(m, param.MAX_ITER); % Safe region radius by iteration
    stop_crit_it = zeros(1, param.MAX_ITER);
    stop_crit_it(1) = inf;

    trace.nb_screen_it = zeros(1,param.MAX_ITER);
    trace.alpha_it = zeros(1,param.MAX_ITER);
    
    %Save all iterates only if not too much memory-demanding (<4GB)
    if m*param.MAX_ITER*8 < 4e9
        x_it = zeros(m, param.MAX_ITER); % Solution estimate by iteration
        x_it(:,1) = x0;
        save_x_it = true;
    else
        x_it = 'Not saved. Too memory-demanding.';
        warning('Not saving all solution iterates, since too memory-demanding.')
        save_x_it = false;
    end
end
if param.save_time
    trace.time_it = zeros(1,param.MAX_ITER); % Elapsed time until end of each iteration
    trace.screen_time_it = zeros(1,param.MAX_ITER); % Elapsed time on screening at each iteration
    trace.time_it(1) = toc(tStart); % time for initializations 
end
if param.calc_gap, trace.gap_it = zeros(1,param.MAX_ITER); end

%% CoD iterations
while (stop_crit > param.TOL) && (k < param.MAX_ITER)
   
    k = k + 1;
    if param.verbose, fprintf('%4d,',k); end

    % Stopping criterion
    if param.calc_gap % Duality Gap
        % Update dual point
        res = y./(Ax+param.epsilon) - 1; %KL case. y - A*x; %LS case
        ATres = A.'*res;
        [theta, ~] = dualUpdateKL(res,ATres,tdual,Atdual);

        primal = f.eval(Ax) + g.eval(x) ;
        if lambda==0
            dual = y(y~=0).'*log(1+theta(y~=0)) - sum(param.epsilon*theta); % since 0*log(a) = 0 for all a>=0. Avoids 0*log(0) = NaN            
        else
            dual = y(y~=0).'*log(1+lambda*theta(y~=0)) - sum(lambda*param.epsilon*theta); % since 0*log(a) = 0 for all a>=0. Avoids 0*log(0) = NaN
        end        
        gap = primal - dual; % gap has to be calculated anyway for GAP_Safe
        trace.gap_it(k) = gap;        
    end
    if strcmp(param.stop_crit, 'gap')        
        stop_crit = gap;
    else %primal variable difference
        stop_crit = norm(x - x_old, 2);
    end
    if param.verbose, stop_crit, end

    % Screening
    if mod(k-2,param.screen_period) == 0, startScreen=tic;
        % Notation: X - S*A
        % Notation: y - A*x
        % R = X./V % But it is later modified
        res = y./(Ax+param.epsilon) - 1;
        ATres = A.'*res; % /!\HEAVY CALCULATION. Not computed by primal update
        [screen_vec, precalc, ~] = nnKLGapSafeScreen(y, A,res,ATres,normA,Atdual,tdual,Ax,precalc);
        if param.save_time, trace.screen_time_it(k) = toc(startScreen); end  %Only screening test time is counted

        % Remove screened coordinates (and corresponding atoms)
        if(any(x(screen_vec)~=0)), Ax = Ax - A(:,screen_vec)*x(screen_vec);end %Update Ax when nonzero entries in x are screened.
        A(:,screen_vec) = []; 
        x(screen_vec) = []; 
        normA(screen_vec) = [];
        Atdual(screen_vec) = [];        
%         precalc.sumA_zero(screen_vec) = [];
    
        rejected_coords(~rejected_coords) = screen_vec;
%         if param.save_time, trace.screen_time_it(k) = toc(startScreen); end  

        trace.nb_screen_it(k) = sum(rejected_coords); 
        trace.alpha_it(k) = precalc.alpha;
        if isfield(options,'oracle_dual'), precalc.oracle_ATtheta(screen_new) = []; end        
    end

    x_old = x + 0; % +0 avoids x_old to be modified within the MEX function
    
    % Update x and Ax(from Hsieh-Dhillon2011 Coord Descent)
    %C implementation. (attention! the value of x is changed inside the MEX function)
    CoD_KL_l1_update(y, A, x, Ax, lambda, param.epsilon, 0); %implemented in C   

    
    % Save intermediate results
    if param.save_all
        % Compute the objective function value if necessary
        if ~param.calc_gap, primal = f.eval(Ax) + g.eval(x); end
        obj(k) =  primal; %f.eval(A*x) + g.eval(x) ;
        % Store screening vector per iteration
%         screen_it(:,k) = screen_vec;
        screen_it(:,k) = rejected_coords;
        % Store iteration values
%         x_it(:, k) = x;
        if save_x_it, x_it(~screen_it(:,k), k) = x; end
        % Store stopping criterion
        stop_crit_it(k) = stop_crit;        
    end
    if param.save_time
        % Store iteration time
        trace.time_it(k) = toc(tStart); % total elapsed time since the beginning 
    end
end


% zero-padding solution
x_old = x;
x = zeros(m,1);
x(~rejected_coords) = x_old;

if param.save_all
    % Trim down stored results
    obj = obj(1:k);
    screen_it = screen_it(:,1:k);
    stop_crit_it = stop_crit_it(1:k);

    trace.nb_screen_it = trace.nb_screen_it(1:k);
    trace.alpha_it = trace.alpha_it(1:k);
    if save_x_it, x_it = x_it(:,1:k); end
else
    x_it = []; obj = []; screen_it = []; stop_crit_it = [];
end
if param.save_time
    trace.time_it = trace.time_it(1:k);
    trace.screen_time_it = trace.screen_time_it(1:k);
else
    trace.time_it = []; 
end
if param.calc_gap, trace.gap_it= trace.gap_it(1:k); end

end
