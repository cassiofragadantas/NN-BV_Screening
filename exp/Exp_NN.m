addpath ../misc/ ../datasets ../
rng_seed = 10; % 0 for no seed
if rng_seed, rng(rng_seed), fprintf('\n\n /!\\/!\\ RANDOM SEED ACTIVATED /!\\/!\\\n\n'); end

%% User-defined parameters  
% Choose dataset
exp_type = 'synthetic'; % Options: 'synthetic', or some real dataset 
                         % Count data: 'NIPSpapers', 
                         %     'MNIST', 'Encyclopedia', '20newsgroups' (n too for big for CoD)
                         %     'TasteProfile'

% Dimensions (only for synthetic experiments, irrelevant otherwise)
m = 1000; %2000
n = 1000; %500, 1000, 2000, 5000
density_x = 0.05;
nb_iter = 30000; %maximum number of iterations

% Algorithm selection (set to false to skip solver) 
MM = false; CoD = true; ActiveSet = true;

% Noise (type and level)
noise_type = 'gaussian_snr'; % Options: 'poisson', 'gaussian_std', 'gaussian_snr', otherwise: no noise.

if strcmp(noise_type,'gaussian_std')
    sigma = 0.1; %noise standard deviation
elseif strcmp(noise_type,'gaussian_snr')
    snr_db = 10;
end
                         

%% Initializations
normalizeA = true;
if strcmp(exp_type,'synthetic')
    A = abs(randn(m,n)); A = A./sqrt(sum(A.^2)); % random A with unit-norm columns
%     y_orig = abs(randn(m,1));
    x_golden = sprand(n,1,density_x); 
    y_orig = A*x_golden;
    y_orig = y_orig/norm(y_orig);
elseif strcmp(exp_type,'cond')
        A = abs(randn(m,n)); A = A./sqrt(sum(A.^2)); % random A with unit-norm columns
        [U, S, V] = svd(A,'econ');
        cond_factor = 1000; % try 10^-3, 10^-1, 1, 10, 1000
        A = U*diag((diag(S)-S(end))*cond_factor + S(end))*V.';
        A = A./sqrt(sum(A.^2));
%     y_orig = abs(randn(m,1));
    x_golden = sprand(n,1,density_x); 
    y_orig = A*x_golden;
    y_orig = y_orig/norm(y_orig);    
elseif strcmp(exp_type,'conv')
    sigma = 20; % filter size ~6 * sigmma + 1
    if n ~= m, disp('Matrix A should be square, setting n = m.'); n=m; end
    h = fspecial('gaussian',[1,2*ceil(3*sigma)+1],sigma); % 1D filter
    halfh = (length(h)-1)/2;
    A = toeplitz([h zeros(1,n-halfh-1)], [h(1) zeros(1,n+halfh-1)]);
    A = A(halfh+1:end,1:end-halfh);
    
    x_golden = sprand(n,1,density_x);
    y_orig = A*x_golden;
else
    load_dataset
%     if normalizeA, y_orig = y_orig/norm(y_orig); end
end

%Add noise
if strcmp(noise_type,'poisson')
    y = poissrnd(full(y_orig));
elseif strcmp(noise_type,'gaussian_std') % gaussian with std
    y = y_orig + sigma*randn(size(y_orig));
elseif strcmp(noise_type,'gaussian_snr') % gaussin with snr
    y = y_orig + (10^(-snr_db/20)*norm(y_orig)/sqrt(m))*randn(m,1);
else % no noise
    y = y_orig;
end
y = max(y,0);
% assert(all(y>=0),'Input signal should only have positive entries.')

% primal and dual cost functions
%Euclidean distance
primal = @(a) 0.5*sum( (y - a).^2 );
dual = @(b) 0.5*sum( y.^2 - (y - b).^2 );    

%Screening initialization
screen_period = 10;
normA = sqrt(sum(A.^2)).';
% sumA = sum(A,1);
%
% Choice of dual translation direction
z = ones(m,1); % -t vector on the paper
% 
% z = sum(A,2); z = z./norm(z); % Take the average of the columns
%
% z = ones(m,1); z(1:floor(m/1.04)) = 0.001; z = z/norm(z); %m/1.1 -> correl 0.3, 1.04 -> 0.2
%
fprintf('Correlation between chosen dual translation direction t and -1 vector: %.5f\n', z.'*ones(m,1)/(sqrt(m)*norm(z)))
sumA = (A.'*z).';
assert(all(sumA>0),'Vector z has to be positively correlated with all columns of A.')
options.tdual = z;

% Storage variables
cost = zeros(1,nb_iter); %save cost function evolution
dualgap = zeros(1,nb_iter); %save cost function evolution
timeIt = zeros(1,nb_iter); %save total time at each iteration
costScreen = zeros(1,nb_iter); %save cost function evolution
dualgapScreen = zeros(1,nb_iter); %save cost function evolution
timeItScreen = zeros(1,nb_iter); %save total time at each iteration
screen_ratio = zeros(1,nb_iter); %save screening ratio evolution
rejected_coords = zeros(1,n);

%% Find x such that y=Ax, given A and y
A0=A;
x0 = abs(randn(n,1)); %random initialization
x0 = zeros(n,1);
%%%%%%%%%%%% MM algorithm %%%%%%%%%%%%
if MM
fprintf('\n======= Majorization-Minimization algorithm =======\n')
% No screening
x = x0;
Ax = A*x;
startTime = tic;
calc_gap = true;
tol = 1e-9*(m/n); deltax = 1; deltax0 = 0;
% for k = 1:nb_iter
while deltax >= tol^2*deltax0 && k <= maxiter %Maximum number of iterations
    xprev = x;
    
    % -- Primal update --
    % Multiplicative update
    % generic beta divergence
%     x = x.*( (A.'*(y.*Ax.^(beta-2))) ./ (A.'*(Ax.^(beta-1))) ); %.^gamma;
    % beta=2 (euclidean case)
    ATy = A.'*y;
    ATAx = A.'*Ax;
    x = x.*( ATy ./ ATAx ); 
    Ax = A*x;
    
    % Stopping criterion
    if k==1, deltax0 = norm(x-x0); else, deltax = norm(x-xprev); end
    
    timeIt(k) = toc(startTime);
    k = k+1;
end
xMM = x;


% Screening
x = x0;
Ax = A*x;
startTime = tic;
tol = 1e-9*(m/n); deltax = 1; deltax0 = 0;
% for k = 1:nb_iter
while deltax >= tol^2*deltax0 && k <= maxiter %Maximum number of iterations
    xprev = x;

    % -- Primal update --
    % Multiplicative update
    ATy = A.'*y;
    ATAx = A.'*Ax;
    x = x.*( ATy ./ ATAx );
    Ax = A*x;

    % -- Screening --
    if mod(k,screen_period) == 0
        % Notation: d - C*X
        [screen_vec, trace] = nnGapSafeScreen(y, A, y-Ax, ATy - ATAx, normA, sumA);
       
        A(:,screen_vec) = []; 
        x(screen_vec) = [];
        sumA(screen_vec) = [];
        normA(screen_vec) = [];
        rejected_coords(~rejected_coords) = screen_vec;       
    end
    
    % Stopping criterion
    if k==1, deltax0 = norm(x-x0); else, deltax = norm(x-xprev); end

    screen_ratio(k) = sum(rejected_coords)/n;
    timeItScreen(k) = toc(startTime);
    
    k = k+1;
end

% zero-padding solution
xprev = x;
x = zeros(n,1);
x(~rejected_coords) = xprev;
A = A0;

% Assert screening did not affect algorithm convergence point
assert(norm(xMM - x)/norm(x)<1e-9, 'Error! Screening changed the MM solver result')

fprintf('MM algorithm : %.2f s\n', timeIt(end))
fprintf('MM + Screening : %.2f s\n', timeItScreen(end))
fprintf('MM speedup : %.2f times \n', timeIt(end)/timeItScreen(end))
end

%%%%%%%%%%%% CoD algorithm %%%%%%%%%%%%
if CoD
fprintf('\n======= Coord. Descent algorithm =======\n')
tic, [xHALS, outHALS]= nnlsHALSupdt(y,A,x0,nb_iter); timeHALS = toc;

tic, [xHALS_screen, outHALS_screen] = nnlsHALS_Screen(y,A,x0,nb_iter,options); timeHALS_Screen = toc;

% Assert screening did not affect algorithm convergence point
assert(norm(xHALS - xHALS_screen)/norm(xHALS_screen)<1e-9, 'Error! Screening changed the CoD solver result')

fprintf('CoD algorithm : %.4s s\n', timeHALS)
fprintf('CoD + Screening : %.4s s\n', timeHALS_Screen)
fprintf('CoD speedup : %.4s times \n', timeHALS/timeHALS_Screen)  

% Re-run to record duality gap at each iteration
fprintf('\n... re-running solvers to compute duality gap offline ...\n')
options.calc_gap = true;
[~, outHALStmp]= nnlsHALSupdt(y,A,x0,nb_iter,options);
[~, outHALS_screentmp] = nnlsHALS_Screen(y,A,x0,nb_iter,options);
options.calc_gap = false;

time1e6 = outHALS.time_it(find(outHALStmp.gap_it<1e-6,1));
time1e6_screen = outHALS_screen.time_it(find(outHALS_screentmp.gap_it<1e-6,1));
fprintf('CoD algorithm : %.4s s (to reach gap<1e-6)\n', time1e6 )
fprintf('CoD + Screening : %.4s s (to reach gap<1e-6)\n', time1e6_screen )
fprintf('CoD speedup : %.4s times \n', time1e6/time1e6_screen)  
end
%%%%%%%%%%%% Active Set algorithm %%%%%%%%%%%%
if ActiveSet
fprintf('\n======= Active Set algorithm =======\n')

% profile on
tic, [xAS,~,~,~,outAS,~]  = lsqnonneg(A,y); timeAS = toc; % x0 is all-zeros
% profile off, profsave(profile('info'),'./new_Profile_AS-NNLS')

% profile on
tic, [xAS_screen,~,~,~,outAS_screen,~] = lsqnonneg_Screen(A,y,options); timeAS_Screen = toc;
% profile off, profsave(profile('info'),'./new_Profile_AS-Screen-NNLS')

% profile on
tic, [xAS_screen2,~,~,~,outAS_screen2,~] = lsqnonneg_Screen2(A,y,options); timeAS_Screen2 = toc;
% profile off, profsave(profile('info'),'./new_Profile_AS-Screen2-NNLS')

% Assert screening did not affect algorithm convergence point
assert(norm(xAS - xAS_screen)/norm(xAS_screen)<1e-9, 'Error! Screening changed the Active Set solver result')
assert(norm(xAS - xAS_screen2)/norm(xAS_screen2)<1e-9, 'Error! Screening changed the Active Set solver result')

fprintf('Active Set algorithm : %.4s s\n', timeAS)
fprintf('Active Set + Screening : %.4s s\n', timeAS_Screen)
fprintf('Active Set + Screening 2: %.4s s\n', timeAS_Screen2)    
fprintf('Active Set speedup : %.4s times \n', timeAS/timeAS_Screen) 
fprintf('Active Set speedup 2: %.4s times \n', timeAS/timeAS_Screen2)  

% Re-run to record duality gap at each iteration
fprintf('\n... re-running solvers to compute duality gap offline ...\n')
options.calc_gap = true;
[~,~,~,~,outAStmp,~]  = lsqnonneg(A,y,options);
[~,~,~,~,outAS_screentmp,~] = lsqnonneg_Screen(A,y,options);
options.calc_gap = false;

time1e6 = outAS.time_it(find(outAStmp.gap_it<1e-6,1));
time1e6_screen = outAS_screen.time_it(find(outAS_screentmp.gap_it<1e-6,1));
fprintf('Active Set algorithm : %.4s s (to reach gap<1e-6)\n', time1e6 )
fprintf('Active Set + Screening : %.4s s (to reach gap<1e-6)\n', time1e6_screen )
fprintf('Active Set speedup : %.4s times \n', time1e6/time1e6_screen)  
end

%% Results
if ~exist('omitResults','var')
    filename = ['new_Exp_NN_' exp_type '_m' num2str(m) 'n' num2str(n) ... 
                '_scrperiod' num2str(screen_period) '_noise-' noise_type ...
                '_sp' num2str(nnz(xAS)/n) '_seed' num2str(rng_seed)];  
            
    % Plot properties setting
    set(0,'DefaultTextInterpreter','latex'), set(0,'DefaultLegendInterpreter','latex');
    set(0,'DefaultAxesTickLabelInterpreter', 'latex'); %groot,
    set(0, 'DefaultLineLineWidth', 2);
    
    legendvec = {}; legendvec2 = {};
    if CoD
    nb_zeros = sum(xHALS<1e-10); % Number of zeros in the solution (baseline)
    %%%% Duality gap vs. Time %%%
    figure(2), subplot(2,1,1),
    % Baseline    
    semilogy(outHALS.time_it,outHALStmp.gap_it,'k'), hold on,
    legendvec{end+1} = 'Coord. Descent (baseline)';
    % Screening
    set(gca,'ColorOrderIndex',1)
    semilogy(outHALS_screen.time_it,outHALS_screentmp.gap_it)
    legendvec{end+1} = 'Coord. Descent + Screening';
    %%%% Screening ratio vs. Time %%%                
    subplot(2,1,2), hold on
    plot([0 outHALS.time_it(end)], repmat(nb_zeros/n,1,2),'--','color',0.5*[1 1 1]) 
    legendvec2{end+1} = 'Oracle';
    set(gca,'ColorOrderIndex',1)
    plot(outHALS_screen.time_it(10:10:end),outHALS_screen.nb_screen_it(10:10:end)/n),
    legendvec2{end+1} = 'Coord. Descent + Screening';
    
    figure(1), hold on, semilogy(outHALS_screen.nb_screen_it(10:10:end)/n)
    end
    
    if ActiveSet
    %%%% Duality gap vs. Time %%%
    figure(2), subplot(2,1,1), grid on
    % Baseline    
    semilogy(outAS.time_it,outAStmp.gap_it,'k:'), hold on,
    legendvec{end+1} = 'Active Set (baseline)';    
    % Screening
    set(gca,'ColorOrderIndex',2)
    semilogy(outAS_screen.time_it,outAS_screentmp.gap_it,':')
    legendvec{end+1} = 'Active Set + Screening';    
    %%%% Screening ratio vs. Time %%%            
    subplot(2,1,2), hold on
%     plot([0 outAS.time_it(end)], repmat(nb_zeros/n,1,2),'--','color',0.5*[1 1 1]) 
    set(gca,'ColorOrderIndex',2)
    plot(outAS_screen.time_it(10:10:end),outAS_screen.nb_screen_it(10:10:end)/n,':'), 
    legendvec2{end+1} = 'Active Set + Screening';    
    end
    
    figure(2), subplot(2,1,1), grid on
    ylabel('Duality gap'), xlabel('Time [s]'), grid on
    legend(legendvec)
    subplot(2,1,2), grid on
%     xlim([0 outHALS.time_it(end)]), ylim([0 1])
    ylabel('Screening ratio [\%]'), xlabel('Time [s]'), grid on
    legend(legendvec2, 'Location', 'southeast') 
    
    savefig([filename '.fig'])
    
    %%%% Screening ratio vs. Iteration number %%%
    if MM
    figure(1)
    subplot(2,1,1), title(['m=' num2str(m) ', n=' num2str(n)]),
    semilogy(dualgapScreen), ylabel('Duality gap')
    legend({ 'MM solver + Screening'})
    subplot(2,1,2), hold on, 
    plot(screen_ratio), ylim([0 1]), ylabel('Screening ratio')
    nb_zeros = sum(xHALS<1e-10); % Number of zeros in the solution (baseline)
    hold on, plot([1 nb_iter], repmat(nb_zeros/n,1,2),'--')
    legend({'Screened (%)', 'Oracle sparsity (%)'}, 'Location', 'southeast')

    %%%% Duality gap vs. Time %%%
    figure(2), subplot(2,1,1), hold on
    semilogy(timeItScreen,dualgapScreen), semilogy(timeIt,dualgap,'k')
    ylabel('Duality gap'), xlabel('Time [s]'), grid on
    legend({ 'MM solver + Screening', 'MM solver'})
    subplot(2,1,2), hold on
    plot(timeItScreen,screen_ratio), xlim([0 timeIt(end)]), ylim([0 1])
    plot([0 timeIt(end)], repmat(nb_zeros/n,1,2),'--','color',0.5*[1 1 1])
    ylabel('Screening ratio'), xlabel('Time [s]'), grid on
    legend({'Screened (%)' 'Oracle (%)'}, 'Location', 'southeast')
    end    
end
