addpath ../solvers/ ../datasets ../utils/ ../
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
oracle_dual = true; % Run screening with oracle dual point

solver = 'PG'; % 'PD' for primal-dual solver. 'PG' for proximal gradient

% Noise (type and level)
noise_type = 'gaussian_std'; % Options: 'poisson', 'gaussian_std', 'gaussian_snr', otherwise: no noise.
%Add noise
if strcmp(noise_type,'gaussian_std')
    sigma = 1; %noise standard deviation
elseif strcmp(noise_type,'gaussian_snr')
    snr_db = 10;
end

%% Initializations
nb_iter = 1e6; %maximum number of iterations

% Data generation (A, y)
if strcmp(exp_type,'synthetic')
    A = abs(randn(m,n)); % gaussian random A
    if normalizeA, A = A./sqrt(sum(A.^2)); end  % unit-norm columns
    % random y
%     y_orig = randn(m,1);
%     y_orig = y_orig/norm(y_orig);
    % y as a combination of columns of A
    density_x = 0.05;
    x_golden = sprand(n,1,density_x); 
    y_orig = A*x_golden;        
    % Lower and Upper bounds (constraints): l <= x <= u 
    l = 0*ones(n,1);
    u = 1*ones(n,1);
elseif strcmp(exp_type,'cond')
    A = abs(randn(m,n));
    if normalizeA, A = A./sqrt(sum(A.^2)); end
    [U, S, V] = svd(A,'econ');
    cond_factor = 100; % try 10^-3, 10^-1, 1, 10, 1000
    A = U*diag((diag(S)-S(end))*cond_factor + S(end))*V.';
    if normalizeA, A = A./sqrt(sum(A.^2)); end
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

x0 = abs(randn(n,1)); %random initialization for solution vector x
L = norm(full(A));
mu = 2/L.^2; % fixed stepsize
calc_gap = true;

%% Find x such that y=Ax, given A and y

%%%%%%%%%%%% Projected gradient (PGD) algorithm %%%%%%%%%%%%
% profile on

if strcmp(solver,'PD')
    fprintf('\n======= Primal-dual algorithm =======\n')
    x0 = zeros(n,1); %randn(n,1); x0 = x0/norm(x0); %random initialization for solution vector x
    [xPGD, outPGD]= ChamPockPGD(y,A,l,u,x0,nb_iter,L,calc_gap);
    [xPGD_screen, outPGD_screen]= ChamPockPGD(y,A,l,u,x0,nb_iter,L,calc_gap,screen_period);
    if oracle_dual
        options.oracle_dual = outPGD_screen.theta; % Oracle dual point
        [xPGD_screenOracle, outPGD_screenOracle]= ChamPockPGD(y,A,l,u,x0,nb_iter,L,calc_gap,screen_period,options);
    end
elseif strcmp(solver,'PG')
    fprintf('\n======= Prox. Grad. algorithm =======\n')
    [xPGD, outPGD]= bvPGD(y,A,l,u,x0,mu,nb_iter,calc_gap);
    [xPGD_screen, outPGD_screen]= bvPGD(y,A,l,u,x0,mu,nb_iter,calc_gap,screen_period);
    if oracle_dual
        options.oracle_dual = y - A*xPGD_screen; % Oracle dual point
        [xPGD_screenOracle, outPGD_screenOracle]= bvPGD(y,A,l,u,x0,mu,nb_iter,calc_gap,screen_period,options);
    end
else
    error('Solver not implemented. Choose either PG or PD solvers.')
end

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
    fprintf([solver ' algorithm : %.2f s\n'], outPGD.timeIt(end))
    fprintf([solver ' + Screening : %.2f s\n'], outPGD_screen.timeIt(end))
    fprintf([solver ' speedup : %.2f times \n\n'], outPGD.timeIt(end)/outPGD_screen.timeIt(end))
    
    if oracle_dual
        fprintf(['\n' solver ' + Screening (oracle dual) : %.2f s\n'], outPGD_screenOracle.timeIt(end))
        fprintf([solver ' speedup : %.2f times \n\n'], outPGD.timeIt(end)/outPGD_screenOracle.timeIt(end))
    end

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
    % Screening
    set(gca,'ColorOrderIndex',1)    
    semilogy(outPGD_screen.timeIt(idx),outPGD_screen.gapIt(idx)),
    legend_arr = {[solver ' (baseline)'], [solver ' + Screening']};
    % Screening (oracle dual)
    if oracle_dual
        set(gca,'ColorOrderIndex',1)    
        semilogy(outPGD_screenOracle.timeIt,outPGD_screenOracle.gapIt,'-.'),
        legend_arr{end+1} = [solver ' + Screening ($\theta^\star$)'];
    end
    % Settings
    ylabel('Duality gap'), xlabel('Time [s]'), grid on
    legend(legend_arr)
    
    %%%% Screening ratio vs. Time %%%        
    subplot(2,1,2), hold on
    % Oracle
    nb_saturated = sum(xPGD>=u-eps) + sum(xPGD<=l+eps);
    plot([0 outPGD.timeIt(end)], repmat(nb_saturated/n,1,2),'--','color',0.5*[1 1 1])
    % Screening
    set(gca,'ColorOrderIndex',1)
    plot(outPGD_screen.timeIt(idx),outPGD_screen.screenIt(idx)), xlim([0 outPGD.timeIt(end)]), ylim([0 1])
    legend_arr = {'Oracle' 'Screened'};
    % Screening (oracle dual)
    if oracle_dual
        set(gca,'ColorOrderIndex',1)
        plot(outPGD_screenOracle.timeIt,outPGD_screenOracle.screenIt,'-.'), xlim([0 outPGD.timeIt(end)]), ylim([0 1])    
        legend_arr{end+1} = 'Screened ($\theta^\star$)';
    end
    % Settings    
    ylabel('Screening ratio'), xlabel('Time [s]'), grid on
    legend(legend_arr, 'Location', 'southeast')
    
    savefig([filename '.fig'])
    end
end

