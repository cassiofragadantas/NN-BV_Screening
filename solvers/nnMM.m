function [x, output] = nnMM(y,A,x,maxiter,calc_gap,screen_period,tdual)

% Default input variables
if nargin < 2, error('nnMM: Inputs {y, A} are mandatory'); end
if nargin < 3, x = ones(size(A,2),1); end
if nargin < 4, maxiter = 1e5; end
if nargin < 5, calc_gap = false; end
if nargin < 6, screen_period = 0; end
if nargin < 7, tdual=ones(size(y)); end
assert(all([x>0; y>0; A(:)>0]),'MM solver: all variables (A,y,x0) should be positive')
[m,n] = size(A);
tol = 1e-9*(m/n);

% Screening initialization (if activated)
if screen_period || calc_gap % Safe screening activated: initializations
    normA = sqrt(sum(A.^2)).';
    sumA = (A.'*tdual).';    
end

% Output variables
output.time_it = zeros(1,maxiter);
if calc_gap, output.gap_it = zeros(1,maxiter); end
if screen_period
    output.nb_screen_it = zeros(1,maxiter);
    rejected_coords = zeros(1,n);
end

% Main loop
startTime = tic;
k = 1; converged = false;
while ~converged && k <= maxiter %Maximum number of iterations
    xprev = x;

    % -- Primal update --
    % Multiplicative update
    Ax = A*x;    
    ATy = A.'*y;
    ATAx = A.'*Ax;
    x = x.*( ATy ./ ATAx );
    % generic beta divergence
%     x = x.*( (A.'*(y.*Ax.^(beta-2))) ./ (A.'*(Ax.^(beta-1))) ); %.^gamma;

    % -- Stopping criterion --
    deltax = norm(x-xprev);
    if k==1, deltax0 = deltax; end
    converged = (deltax < tol^2*deltax0);

    % Not executed normally! Compute gap for illustration-purpose only
    if calc_gap
        [~, trace] = nnGapSafeScreen(y, A, y-Ax, ATy - ATAx, normA, sumA,tdual);
        output.gap_it(k) = trace.gap;
    end

    % -- Screening --
    if mod(k,screen_period) == 0
        [screen_vec, ~] = nnGapSafeScreen(y, A, y-Ax, ATy - ATAx, normA, sumA,tdual);
       
        A(:,screen_vec) = []; 
        x(screen_vec) = [];
        sumA(screen_vec) = [];
        normA(screen_vec) = [];
        rejected_coords(~rejected_coords) = screen_vec;
        output.nb_screen_it(k) = sum(rejected_coords);
    end

    output.time_it(k) = toc(startTime);
    
    k = k+1;
end

% Trimming output variables
output.time_it = output.time_it(1:k-1);
if screen_period, output.nb_screen_it = output.nb_screen_it(1:k-1); end
if calc_gap, output.gap_it= output.gap_it(1:k-1); end

% zero-padding solution (if screening activated)
if screen_period
    xprev = x;
    x = zeros(n,1);
    x(~rejected_coords) = xprev;
end