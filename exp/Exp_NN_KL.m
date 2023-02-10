addpath ../solvers/ ../solvers/ASNA/ImprovedMatlab/ ../solvers/ASNA/OriginalMatlab/  ../datasets ../utils ../
rng_seed = 10; % 0 for no seed
if rng_seed, rng(rng_seed); end
omitResults = true; % Comment to display results

%% User-defined parameters  
% Choose dataset
exp_type = 'synthetic'; % Options: 'synthetic', or some real dataset 
                         % Count data: 'NIPSpapers', 
                         %     'MNIST', 'Encyclopedia', '20newsgroups' (n too for big for CoD)
                         %     'TasteProfile'

% Dimensions (only for synthetic experiments, irrelevant otherwise)
m = 5000; %2000
n = 5000; %500, 1000, 2000, 5000
density_x = 0.1;
nb_iter = 1000; %maximum number of iterations
screen_period = 10;

% Solver selection (set to false to skip) 
MM = false; PG = false; ActiveSet = true;

% Noise (type and level)
noise_type = 'None'; % Options: 'poisson', 'gaussian_std', 'gaussian_snr', otherwise: no noise.
noise_val = 0; % snr or noise standard deviation

% Generate data
[A,y,n,tdual] = genData(m,n,density_x,exp_type,noise_type,noise_val);

% Solvers initialization
x0 = ones(n,1); % abs(randn(n,1));

%% Find x such that y=Ax, given A and y
%%%%%%%%%%%% MM algorithm %%%%%%%%%%%%
if MM    
    fprintf('\n======= Majorization-Minimization algorithm =======\n')
    error('NOT IMPLEMENTED YET !!!!')
    % Run solvers
    tic, [xMM, outMM]= nnMM(y,A,x0,nb_iter); timeMM = toc;
    tic, [xMM_screen, outMM_screen] = nnMM(y,A,x0,nb_iter,false,screen_period,tdual); timeMM_Screen = toc;
    
    % Assert screening did not affect algorithm convergence point
    assert(norm(xMM - xMM_screen)/norm(xMM_screen)<1e-9, 'Error! Screening changed the MM solver result')
    
    print_time('MM',timeMM,timeMM_Screen, false)
    
    % Re-run to record duality gap at each iteration
    fprintf('\n... re-running solvers to compute duality gap offline ...\n')
    [~, outMMtmp]= nnMM(y,A,x0,nb_iter,true,0,tdual);
    [~, outMM_screentmp] = nnMM(y,A,x0,nb_iter,true,screen_period,tdual);
    
    time1e6MM = outMM.time_it(find(outMMtmp.gap_it<1e-6,1));
    time1e6MM_screen = outMM_screen.time_it(find(outMM_screentmp.gap_it<1e-6,1));
    print_time('MM',time1e6MM,time1e6MM_screen, true)
end

%%%%%%%%%%%% Prox. Grad. algorithm %%%%%%%%%%%%
if PG
    fprintf('\n======= Prox. Gradient algorithm =======\n')
    error('NOT IMPLEMENTED YET !!!!')
    options.screen_period = screen_period;
    options.tdual = tdual;

    % Run solvers
    tic, [xHALS, outHALS]= nnlsHALSupdt(y,A,x0,nb_iter); timeHALS = toc;
    tic, [xHALS_screen, outHALS_screen] = nnlsHALS_Screen(y,A,x0,nb_iter,options); timeHALS_Screen = toc;

    % Assert screening did not affect algorithm convergence point
    assert(norm(xHALS - xHALS_screen)/norm(xHALS_screen)<1e-9, 'Error! Screening changed the CoD solver result')
    
    print_time('CoD',timeHALS,timeHALS_Screen, false)  
    
    % Re-run to record duality gap at each iteration
    fprintf('\n... re-running solvers to compute duality gap offline ...\n')
    options.calc_gap = true;
    [~, outHALStmp]= nnlsHALSupdt(y,A,x0,nb_iter,options);
    [~, outHALS_screentmp] = nnlsHALS_Screen(y,A,x0,nb_iter,options);
    options.calc_gap = false;
    
    time1e6HALS = outHALS.time_it(find(outHALStmp.gap_it<1e-6,1));
    time1e6HALS_screen = outHALS_screen.time_it(find(outHALS_screentmp.gap_it<1e-6,1));
    print_time('CoD',time1e6HALS,time1e6HALS_screen, true)
end

%%%%%%%%%%%% Active Set algorithm %%%%%%%%%%%%
if ActiveSet
    fprintf('\n======= Active Set algorithm =======\n')
    options.screen_period = screen_period;
    options.tdual = tdual;

    % Run solvers
%     profile on
    tic, [xAS, outAS] = asnaMine(y,A,0,nb_iter); timeAS = toc; % x0 defined inside
%     tic, xAS = asna(y.',A.',0,nb_iter); timeAS =      toc; xAs = xAS.'; % x0 defined inside    
%     profile off, profsave(profile('info'),'./new_Profile_AS-NNKL')

%     profile on
    tic, [xAS_screen, outAS_screen]=asnaMine_Screen(y,A,0,nb_iter,options); timeAS_Screen = toc;    
%     profile off, profsave(profile('info'),'./new_Profile_AS-Screen-NNKL')
    
    % Assert screening did not affect algorithm convergence point
%     assert(norm(xAS - xAS_screen)/norm(xAS_screen)<1e-9, 'Error! Screening changed the Active Set solver result')

    print_time('Active Set',timeAS,timeAS_Screen, false)  
    
    % Re-run to record duality gap at each iteration
    fprintf('\n... re-running solvers to compute duality gap offline ...\n')
    options.calc_gap = true;
    [~, outAStmp] = asnaMine(y,A,0,nb_iter,options);
    [~, outAS_screentmp] = asnaMine_Screen(y,A,0,nb_iter,options);
    options.calc_gap = false;
    
%     time1e6AS = outAS.time_it(find(outAStmp.gap_it<1e-6,1));
    time1e6AS_screen = outAS_screen.time_it(find(outAS_screentmp.gap_it<1e-6,1));
%     print_time('Active Set',time1e6AS,time1e6AS_screen, true)
end

%% Results
if ~exist('omitResults','var')
            
    % Plot properties setting
    set(0,'DefaultTextInterpreter','latex'), set(0,'DefaultLegendInterpreter','latex');
    set(0,'DefaultAxesTickLabelInterpreter', 'latex'); %groot,
    set(0, 'DefaultLineLineWidth', 2);
    
    max_time = 0; max_iter = 0;
    legendvec = {}; legendvec2 = {}; legendvec3 = {};
    clear nb_zeros

    if CoD
        [leg, leg2, leg3] = plot_results('Coord. Descent', n, ...
                1, outHALS, outHALS_screen, outHALStmp, outHALS_screentmp, screen_period);
        legendvec = [legendvec leg{:}]; legendvec2 = [legendvec2 leg2{:}]; legendvec3 = [legendvec3 leg3{:}];
        nb_zeros = sum(xHALS<1e-10); % Number of zeros in the solution (baseline)
        max_time = max(max_time,outHALS.time_it(end));
        max_iter = max(max_iter,length(outHALS.time_it));
        %%%% Identifiable zeros %%%%
        res = (y - A*xHALS_screen);
        ATres = A.'*res;
        sumA = tdual.'*A;    
        epsilon = max(ATres./sumA.'); %max(ATres) also works if A is normalized, but is slightly worse.
        %theta = res - epsilon*tdual;
        ATtheta = ATres - epsilon*sumA.'; %= A.'* theta; Should be zero at coordinates xj ~= 0 and negative otherwise
        screenable_zeros = (ATtheta<-sqrt(2*outHALS_screentmp.gap_it(end)));
    end
    
    if ActiveSet
        [leg, leg2, leg3] = plot_results('Active Set', n, ...
                2, outAS, outAS_screen, outAStmp, outAS_screentmp, screen_period);
        legendvec = [legendvec leg{:}]; legendvec2 = [legendvec2 leg2{:}]; legendvec3 = [legendvec3 leg3{:}];
        if ~exist('nb_zeros','var'),nb_zeros = sum(xAS<1e-10); end
        max_time = max(max_time,outAS.time_it(end));
        max_iter = max(max_iter,length(outAS.time_it));
    end

    if MM
        [leg, leg2, leg3] = plot_results('Multiplicative', n, ...
                3, outMM, outMM_screen, outMMtmp, outMM_screentmp, screen_period);
        legendvec = [legendvec leg{:}]; legendvec2 = [legendvec2 leg2{:}]; legendvec3 = [legendvec3 leg3{:}];
        if ~exist('nb_zeros','var'),nb_zeros = sum(xMM<1e-10); end
        max_time = max(max_time,outMM.time_it(end));
        max_iter = max(max_iter,length(outMM.time_it));
    end        

    % Oracle screening ratio
    figure(2), subplot(2,1,2)
    plot([0 max_time], repmat(nb_zeros/n,1,2),'--','color',0.5*[1 1 1]) 
    legendvec2{end+1} = 'Oracle';
    figure(1), hold on
    plot([0 max_iter], repmat(nb_zeros/n,1,2),'--','color',0.5*[1 1 1]) 
    legendvec3{end+1} = 'Oracle';

    % Labels, title, legend
    figure(1), title(['m=' num2str(m) ', n=' num2str(n)]),
    ylabel('Screening ratio'), xlabel('Iteration number'), grid on
    legend(legendvec3, 'Location', 'southeast')
    
    figure(2), subplot(2,1,1), grid on
    title([exp_type ': m=' num2str(m) ', n=' num2str(n)]),
    ylabel('Duality gap'), xlabel('Time [s]'), grid on
    legend(legendvec)
    subplot(2,1,2), grid on
    ylabel('Screening ratio [\%]'), xlabel('Time [s]'), grid on
    legend(legendvec2, 'Location', 'southeast') 

    % Save figure
    filename = ['new_Exp_NN_' exp_type '_m' num2str(m) 'n' num2str(n) ... 
            '_scrperiod' num2str(screen_period) '_noise-' noise_type ...
            '_sp' num2str(nb_zeros/n) '_seed' num2str(rng_seed)];  
    savefig([filename '.fig'])

end

function print_time(solver_name, time_base, time_screen, gap)

fprintf([solver_name ' algorithm   : %.4s s'], time_base)
if gap, fprintf(' (to reach gap<1e-6)\n'), else, fprintf('\n'), end
fprintf([solver_name ' + screening : %.4s s\n'], time_screen)
fprintf([solver_name ' speedup     : %.4s times \n'], time_base/time_screen)

end

function [legendvec, legendvec2, legendvec3] = plot_results(solver_name, n, ...
            color, out, out_screen, out_gap, out_screen_gap, screen_period)

    legendvec = {}; legendvec2 = {}; legendvec3 = {};

    %%%% Duality gap vs. Time %%%
    figure(2), subplot(2,1,1),
    % Baseline
    set(gca,'ColorOrderIndex',color)
    semilogy(out.time_it,out_gap.gap_it,':'), hold on,
    legendvec{end+1} = [solver_name ' (baseline)'];
    % Screening
    set(gca,'ColorOrderIndex',color)
    semilogy(out_screen.time_it,out_screen_gap.gap_it)
    legendvec{end+1} = [solver_name ' + Screening'];
    %%%% Screening ratio vs. Time %%%
    subplot(2,1,2), hold on
    set(gca,'ColorOrderIndex',color)
    plot(out_screen.time_it(screen_period:screen_period:end),out_screen.nb_screen_it(screen_period:screen_period:end)/n),
    legendvec2{end+1} = [solver_name ' + Screening'];

    %%%% Screening ratio vs. Iteration number %%%
    figure(1), hold on,
    set(gca,'ColorOrderIndex',color)
    semilogy(screen_period:screen_period:length(out_screen.nb_screen_it), ...
             out_screen.nb_screen_it(screen_period:screen_period:end)/n)
    legendvec3{end+1} = [solver_name ' + Screening'];

end
