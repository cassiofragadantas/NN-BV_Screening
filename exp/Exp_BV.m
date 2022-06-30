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
m = 1000;
n = 500;
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
x0 = abs(randn(n,1)); %random initialization for solution vector x

% Data generation (A, y)
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

if strcmp(noise_type,'poisson')
    y = poissrnd(full(y_orig));
elseif strcmp(noise_type,'gaussian_std') % gaussian with std
    y = y_orig + sigma*randn(size(y_orig));
elseif strcmp(noise_type,'gaussian_snr') % gaussin with snr
    y = y_orig + (10^(-snr_db/20)*norm(y_orig)/sqrt(m))*randn(m,1);
else % no noise
    y = y_orig;
end

mu = 2/norm(A).^2; % fixed stepsize
calc_gap = true;

%% Find x such that y=Ax, given A and y
% profile on

%%%%%%%%%%%% Projected gradient (PGD) algorithm %%%%%%%%%%%%
% profile on

fprintf('\n======= Coord. Descent algorithm =======\n')
[xPGD, outPGD]= bvPGD(y,A,l,u,x0,mu,nb_iter,calc_gap);
[xPGD_screen, outPGD_screen]= bvPGD(y,A,l,u,x0,mu,nb_iter,calc_gap,screen_period);

% Assert screening did not affect algorithm convergence point
assert(norm(xPGD - xPGD_screen)<=1e-5*norm(xPGD_screen), ...
       'Warning! Screening seems to have changed the PGD solver result')

% profile off, profsave(profile('info'),'./new_Profile_PG_BVLS')

%% Results
filename = ['new_Exp_BV_' exp_type '_m' num2str(m) 'n' num2str(n) ... 
            '_scrperiod' num2str(screen_period) '_noise-' noise_type ...
            '_sp' num2str(nnz(xPGD_screen)/n) '_seed' num2str(rng_seed)];
        
if ~exist('omitResults','var')
    %%%% Show times %%%%
    fprintf('PG algorithm : %.2f s\n', outPGD.timeIt(end))
    fprintf('PG + Screening : %.2f s\n', outPGD_screen.timeIt(end))
    fprintf('PG speedup : %.2f times \n\n', outPGD.timeIt(end)/outPGD_screen.timeIt(end))
    
    if calc_gap
    % Plot properties setting
    set(0,'DefaultTextInterpreter','latex'), set(0,'DefaultLegendInterpreter','latex');
    set(0,'DefaultAxesTickLabelInterpreter', 'latex'); %groot,
    set(0, 'DefaultLineLineWidth', 2);
        
    figure(1)
    k = min(length(outPGD.timeIt),length(outPGD_screen.timeIt));
    idx = round(linspace(1,k,min(1000,k))); % 1:k
    %%%% Duality gap vs. Time %%%       
    subplot(2,1,1)
    % Baseline
    semilogy(outPGD.timeIt(idx),outPGD.gapIt(idx),'k'),  hold on,
    set(gca,'ColorOrderIndex',1)
    % Screening
    semilogy(outPGD_screen.timeIt(idx),outPGD_screen.gapIt(idx)), 
    % Settings
    ylabel('Duality gap'), xlabel('Time [s]'), grid on
    legend({'Prox. Grad. (baseline)', 'Prox. Grad + Screening', })
    
    %%%% Screening ratio vs. Time %%%        
    subplot(2,1,2), hold on
    % Oracle
    nb_saturated = sum(xPGD>=u-eps) + sum(xPGD<=l+eps);
    plot([0 outPGD.timeIt(end)], repmat(nb_saturated/n,1,2),'--','color',0.5*[1 1 1])
    % Screening
    set(gca,'ColorOrderIndex',1)
    plot(outPGD_screen.timeIt(idx),outPGD_screen.screenIt(idx)), xlim([0 outPGD.timeIt(end)]), ylim([0 1])
    % Settings    
    ylabel('Screening ratio [\%]'), xlabel('Time [s]'), grid on
    legend({'Oracle' 'Screened' }, 'Location', 'southeast')
    
    savefig([filename '.fig'])
    end
end

