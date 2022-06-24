addpath ../misc/ ../datasets ../
rng_seed = 100; % 0 for no seed
if rng_seed, rng(rng_seed), fprintf('\n\n /!\\/!\\ RANDOM SEED ACTIVATED /!\\/!\\\n\n'); end

%% User-defined parameters  
% Choose dataset
exp_type = 'synthetic'; % Options: 'synthetic', or some real dataset 
                         % Hyperspectral : 
                         %     'Urban', 'Urban_subsampled'
                         %     'Cuprite', 'Cuprite_subsampled', 'Moffett', 'Madonna'
                         %     'Cuprite_USGS-lib', 'Urban_USGS-lib' (using USGS spectral library as dictionary)

% Dimensions (only for synthetic experiments, irrelevant otherwise)
m = 2000;
n = 1000;
screen_period = 10;
normalizeA = true;

% Convergence tolerance
gap_tol = 1e-6;

% Noise (type and level)
noise_type = 'none'; % Options: 'poisson', 'gaussian_std', 'gaussian_snr', otherwise: no noise.
%Add noise
if strcmp(noise_type,'gaussian_std')
    sigma = 0.1; %noise standard deviation
elseif strcmp(noise_type,'gaussian_snr')
    snr_db = 10;
end

%% Initializations
nb_iter = 1e7; %maximum number of iterations

if strcmp(exp_type,'synthetic')
    A = abs(randn(m,n)); A = A./sqrt(sum(A.^2)); % random A with unit-norm columns
    % random y
    y_orig = randn(m,1);
    y_orig = y_orig/norm(y_orig);
    % y as a combination of columns of A
%     density_x = 0.05;
%     x_golden = sprand(n,1,density_x); 
%     y_orig = A*x_golden;        
    % Lower and Upper bounds (constraints): l <= x <= u 
    l = 0*ones(n,1);
    u = 1*ones(n,1);
elseif strcmp(exp_type,'cond')
    A = abs(randn(m,n)); A = A./sqrt(sum(A.^2)); % random A with unit-norm columns
    [U, S, V] = svd(A,'econ');
    cond_factor = 100; % try 10^-3, 10^-1, 1, 10, 1000
    A = U*diag((diag(S)-S(end))*cond_factor + S(end))*V.';
    A = A./sqrt(sum(A.^2));
    y_orig = abs(randn(m,1)); y_orig = y_orig/norm(y_orig);
%     x_golden = sprand(n,1,density_x); 
%     y_orig = A*x_golden;
%     y_orig = y_orig/norm(y_orig);
    % Lower and Upper bounds (constraints): l <= x <= u 
    l = 0*ones(n,1);
    u = 1*ones(n,1);   
else
    load_dataset
    % Synthetic y (combining columns of A)
%     density_x = 0.05;
%     x_golden = sprand(n,1,density_x); 
%     y_orig = A*x_golden;
%     y_orig = y_orig/norm(y_orig); 
    % Lower and Upper bounds (constraints): l <= x <= u 
    l = 0*ones(n,1);
    u = 1*ones(n,1);
end


mu = 2/norm(A).^2; % fixed stepsize

if strcmp(noise_type,'poisson')
    y = poissrnd(full(y_orig));
elseif strcmp(noise_type,'gaussian_std') % gaussian with std
    y = y_orig + sigma*randn(size(y_orig));
elseif strcmp(noise_type,'gaussian_snr') % gaussin with snr
    y = y_orig + (10^(-snr_db/20)*norm(y_orig)/sqrt(m))*randn(m,1);
else % no noise
    y = y_orig;
end

beta = 2;
% primal and dual cost functions
if beta == 1
    %KL divergence
    epsAx = 0;
    primal = @(a) sum(v .* log(y ./ (a+epsAx)),'omitnan') + sum(- y + a + epsAx); % 0*log0 = 0
    dual = @(b, Atb) sum(y.*log(1+b),'omitnan') - sum(epsAx*b) - l.'*max(0,Atb) - u.'*min(0,Atb); % force 0*log(0) = 0 instead of NaN
elseif beta == 2
    %Euclidean distance (beta = 2)
    primal = @(a) 0.5*sum( (y - a).^2 );
    dual = @(b,Atb) 0.5*sum( y.^2 - (y - b).^2 ) - l.'*min(0,Atb) - u.'*max(0,Atb);    
    primalScreen = @(yvec, a) 0.5*sum( (yvec - a).^2 );
    dualScreen = @(yvec,b,Atb,lvec,uvec) 0.5*sum( yvec.^2 - (yvec - b).^2 ) - lvec.'*min(0,Atb) - uvec.'*max(0,Atb);
elseif beta == 1.5
    %Beta divergence beta = 1.5
    epsAx = 0;    
    primal = @(a) 4/3 * sum( (y).^1.5 + (1/2)*(a + epsAx).^(1.5) ...
              - (3/2)*(y).*sqrt(a + epsAx)); %beta divergence      
    dual = @(b, Atb) sum( (4*y.^1.5 + 0.5*b.^3 - 0.5*(b.^2 + 4*y).^1.5 + 3*b.*y)/3 - epsAx*b ) ...
                - l.'*max(0,Atb) - u.'*min(0,Atb);
else
    error('Value of beta not implemented!')
end

%Screening initialization
if beta == 2
    normA = sqrt(sum(A.^2)).';
else
    error('Screening initialization not implemented for this value of beta')
end

% Storage variables
cost = zeros(1,nb_iter); %save cost function evolution
dualgap = zeros(1,nb_iter); %save cost function evolution
timeIt = zeros(1,nb_iter); %save total time at each iteration
costScreen = zeros(1,nb_iter); %save cost function evolution
dualgapScreen = zeros(1,nb_iter); %save cost function evolution
timeItScreen = zeros(1,nb_iter); %save total time at each iteration
screen_ratio = zeros(1,nb_iter); %save screening ratio evolution
saturated_coords = zeros(1,n); % saturated coords

%% Find x such that y=Ax, given A and y
% profile on
x0 = abs(randn(n,1)); %random initialization


% No screening
x = x0;
Ax = A*x;
startTime = tic;
k=1; 
gap = inf; calc_gap = true;
delta_tol = 1e-8; delta_x = 1; delta_x0 = 0;
while gap >= gap_tol && k <= nb_iter
% while delta_x >= delta_tol*delta_x0 && k <= nb_iter
    xprev = x;
    % -- Primal update --
    if beta == 2
        % Projected gradient
        res = y - Ax;
        grad = -A.'*res;
        x = x - mu*grad;
        x(x<l) = l(x<l); x(x>u) = u(x>u); % projection step (saturate entries)
        Ax = A*x;     
    else
        res = (y - Ax).*Ax.^(beta-2);
        grad = -A.'*res;
        error('Primal update not implemented for this beta')
    end
    
    % -- Stopping criterion --
%     delta_x = norm(x-xprev)^2;
%     if k == 1, delta_x0 = delta_x; end

    if calc_gap
        % -- Dual update --
        theta =  res; % simply the (generalized) residual
        ATtheta = -grad;

        % -- Duality gap -- 
        cost(k) =  primal(Ax);
        gap = cost(k) - dual(theta,ATtheta); % gap has to be calculated anyway for GAP_Safe
        gap(gap<=0) = eps;
        dualgap(k) = gap;    
    end
    timeIt(k) = toc(startTime);
    k = k+1;
end
xBase = x;
cost = cost(1:k-1);
timeIt = timeIt(1:k-1);
dualgap = dualgap(1:k-1);


% Screening
A0=A;
u0 = u; l0=l;
x = x0;
Ax = A*x;
startTime = tic;
k=1; gap = inf;
while gap >= gap_tol && k <= nb_iter 
    % -- Primal update --
    if beta == 2
        % Projected gradient
        res = y - Ax;
        grad = -A.'*res;
        x = x - mu*grad;
        x(x<l) = l(x<l); x(x>u) = u(x>u); % projection step (saturate entries)
        Ax = A*x;
    else
        res = (y - Ax).*Ax.^(beta-2);
        grad = -A.'*res;
        error('Primal update not implemented for this beta')
    end

    % -- Dual update --
    theta =  res; % simply the (generalized) residual
    ATtheta = -grad;

    % -- Stopping criterion -- 
    costScreen(k) =  primalScreen(y,Ax);
    gap = costScreen(k) - dualScreen(y,theta,ATtheta,l,u); % gap has to be calculated anyway for GAP_Safe
    gap(gap<=0) = eps;
    dualgapScreen(k) = gap;
    
    % Screening
    if mod(k,screen_period) == 0
        if beta == 2
            radius = sqrt(2*gap);
            screen_vec_l = (ATtheta + radius*normA < 0);
            screen_vec_u = (ATtheta - radius*normA > 0);
        else
            error('Screening not implemented for this beta.')
        end
        y = y - A(:,screen_vec_l)*l(screen_vec_l) - A(:,screen_vec_u)*u(screen_vec_u);
        Ax = Ax - A(:,screen_vec_l)*x(screen_vec_l) - A(:,screen_vec_u)*x(screen_vec_u);    
        A(:,screen_vec_l | screen_vec_u) = [];    
        x(screen_vec_l | screen_vec_u) = []; 
        normA(screen_vec_l | screen_vec_u) = [];
        l(screen_vec_l | screen_vec_u) = [];
        u(screen_vec_l | screen_vec_u) = [];
        saturated_coords(~saturated_coords) = screen_vec_u - screen_vec_l;  
    end
    screen_ratio(k) = sum(abs(saturated_coords))/n;

    timeItScreen(k) = toc(startTime);
    k = k+1;
end
k = k-1;
timeItScreen = timeItScreen(1:k);
costScreen = costScreen(1:k);
dualgapScreen = dualgapScreen(1:k);
screen_ratio = screen_ratio(1:k);


% zero-padding solution
x_old = x;
x = zeros(n,1);
x(~saturated_coords) = x_old;
x(saturated_coords == 1) = u0(saturated_coords == 1);
x(saturated_coords == -1) = l0(saturated_coords == -1);
A = A0;

% Assert screening did not affect algorithm convergence point
assert(norm(xBase - x)<=gap_tol*norm(x), 'Error! Screening changed the PG solver result')

% profile off, profsave(profile('info'),'./new_Profile_PG_BVLS')

%% Results
filename = ['new_Exp_BV_' exp_type '_m' num2str(m) 'n' num2str(n) ... 
            '_scrperiod' num2str(screen_period) '_noise-' noise_type ...
            '_sp' num2str(nnz(x)/n) '_seed' num2str(rng_seed)];
        
if ~exist('omitResults','var')
    %%%% Show times %%%%
    fprintf('PG algorithm : %.2f s\n', timeIt(end))
    fprintf('PG + Screening : %.2f s\n', timeItScreen(end))
    fprintf('PG speedup : %.2f times \n\n', timeIt(end)/timeItScreen(end))
    
    if calc_gap
    % Plot properties setting
    set(0,'DefaultTextInterpreter','latex'), set(0,'DefaultLegendInterpreter','latex');
    set(0,'DefaultAxesTickLabelInterpreter', 'latex'); %groot,
    set(0, 'DefaultLineLineWidth', 2);
        
    figure(1)
    idx = round(linspace(1,k,min(1000,k))); % 1:k
    %%%% Duality gap vs. Time %%%       
    subplot(2,1,1)
    % Baseline
    semilogy(timeIt(idx),dualgap(idx),'k'),  hold on,
    set(gca,'ColorOrderIndex',1)
    % Screening
    semilogy(timeItScreen(idx),dualgapScreen(idx)), 
    % Settings
    ylabel('Duality gap'), xlabel('Time [s]'), grid on
    legend({'Prox. Grad. (baseline)', 'Prox. Grad + Screening', })
    
    %%%% Screening ratio vs. Time %%%        
    subplot(2,1,2), hold on
    % Oracle
    nb_saturated = sum(xBase>=u0-eps) + sum(xBase<=l0+eps);
    plot([0 timeIt(end)], repmat(nb_saturated/n,1,2),'--','color',0.5*[1 1 1])
    % Screening
    set(gca,'ColorOrderIndex',1)
    plot(timeItScreen(idx),screen_ratio(idx)), xlim([0 timeIt(end)]), ylim([0 1])
    % Settings    
    ylabel('Screening ratio [\%]'), xlabel('Time [s]'), grid on
    legend({'Oracle' 'Screened' }, 'Location', 'southeast')
    
    savefig([filename '.fig'])
    end
end

