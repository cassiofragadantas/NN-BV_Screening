addpath ../misc/ ../datasets ../utils ../
rng_seed = 10; % 0 for no seed
if rng_seed, rng(rng_seed); end

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
noise_val = 10; % snr or noise standard deviation

% Generate data
[A,y,options.tdual] = genData(m,n,density_x,exp_type,noise_type,noise_val);

%% Initializations

% Primal and dual cost functions
% Euclidean distance
primal = @(a) 0.5*sum( (y - a).^2 );
dual = @(b) 0.5*sum( y.^2 - (y - b).^2 );    

% Screening initialization
screen_period = 10;
normA = sqrt(sum(A.^2)).';
sumA = (A.'*options.tdual).';

% Storage variables
cost = zeros(1,nb_iter);            %save cost function evolution
dualgap = zeros(1,nb_iter);         %save duality gap evolution
timeIt = zeros(1,nb_iter);          %save total time at each iteration
costScreen = zeros(1,nb_iter);      %save cost function evolution
dualgapScreen = zeros(1,nb_iter);   %save duality evolution
timeItScreen = zeros(1,nb_iter);    %save total time at each iteration
screen_ratio = zeros(1,nb_iter);    %save screening ratio evolution
rejected_coords = zeros(1,n);

%% Find x such that y=Ax, given A and y
A0=A; 
x0 = zeros(n,1); % abs(randn(n,1));
%%%%%%%%%%%% MM algorithm %%%%%%%%%%%%
if MM
fprintf('\n======= Majorization-Minimization algorithm =======\n')
assert(all([x0>0; y>0; A(:)>0]),'MM solver: all variables (A,y,x0) should be positive')
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


