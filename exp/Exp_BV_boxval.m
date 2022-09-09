addpath ../ ../solvers/ ../utils/
rng_seed = 10; % 0 for no seed
if rng_seed, rng(rng_seed), fprintf('\n\n /!\\/!\\ RANDOM SEED ACTIVATED /!\\/!\\\n\n'); end

m = 4000;
n = 2000;
beta = 2;
nb_iter = 100000;
gap_tol = 1e-6;
screen_period = 10;
x0 = abs(randn(n,1)); %random initialization

box_val = logspace(-1,log10(5),10); %[0.1 0.5 1 5];
screen_speedup = zeros(size(box_val));
saturated_ratio = zeros(size(box_val));

for k_box = 1:length(box_val)
k_box

% l <= x <= u 
l = -box_val(k_box)*ones(n,1);
u = box_val(k_box)*ones(n,1);

A = randn(m,n); A = A./sqrt(sum(A.^2)); % random A with unit-norm columns
% A = [A eye(m)]; n=n+m;
y = randn(m,1);

mu = 1/norm(A)^2; % fixed stepsize

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
    precalc.normA = sqrt(sum(A.^2)).';
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

% No screening
x = x0;
Ax = A*x;
startTime = tic;
calc_gap = true;
%for k = 1:nb_iter
gap= inf; k = 1;
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

    if calc_gap
        % -- Dual update --
        theta =  res; % simply the (generalized) residual
        ATtheta = -grad;

        % -- Stopping criterion -- 
        cost(k) =  primal(Ax);
        gap = cost(k) - dual(theta,ATtheta); % gap has to be calculated anyway for GAP_Safe
        gap(gap<=0) = eps;
        dualgap(k) = gap;    
    end
    timeIt(k) = toc(startTime);
    k = k+1;

end
time = toc(startTime);
xBase = x;

% Screening
A0=A;
u0 = u; l0=l;
x = x0;
Ax = A*x;
startTime = tic;
%for k = 1:nb_iter
gap = inf; k = 1;
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
            screen_vec_l = (ATtheta + radius*precalc.normA < 0);
            screen_vec_u = (ATtheta - radius*precalc.normA > 0);
        else
            error('Screening not implemented for this beta.')
        end
        y = y - A(:,screen_vec_l)*l(screen_vec_l) - A(:,screen_vec_u)*u(screen_vec_u);
        Ax = Ax - A(:,screen_vec_l)*x(screen_vec_l) - A(:,screen_vec_u)*x(screen_vec_u);    
        A(:,screen_vec_l | screen_vec_u) = [];    
        x(screen_vec_l | screen_vec_u) = []; 
        precalc.normA(screen_vec_l | screen_vec_u) = [];
        l(screen_vec_l | screen_vec_u) = [];
        u(screen_vec_l | screen_vec_u) = [];
        saturated_coords(~saturated_coords) = screen_vec_u - screen_vec_l;  
    end
    screen_ratio(k) = sum(abs(saturated_coords))/n;

    timeItScreen(k) = toc(startTime);
    k = k+1;

end
timeScreen = toc(startTime);

% zero-padding solution
x_old = x;
x = zeros(n,1);
x(~saturated_coords) = x_old;
x(saturated_coords == 1) = u0(saturated_coords == 1);
x(saturated_coords == -1) = l0(saturated_coords == -1);
A = A0;

nb_saturated = sum(xBase>=u0-eps) + sum(xBase<=l0+eps);
saturated_ratio(k_box) = nb_saturated/n
screen_speedup(k_box) = time(end)/timeScreen(end)
end
%% Results
if ~exist('omitResults','var')

% Plot properties setting
set(0,'DefaultTextInterpreter','latex'), set(0,'DefaultLegendInterpreter','latex');
set(0,'DefaultAxesTickLabelInterpreter', 'latex'); %groot,
set(0, 'DefaultLineLineWidth', 2);
    
figure(1)
xlabel('Box limits b*[-1,1]')
yyaxis right
plot(box_val,saturated_ratio,'LineWidth',1.5)
ylabel('Saturation ratio [\%]')
grid on, ylim([-0.1 1])
figure(1), yyaxis left
plot(box_val,screen_speedup,'LineWidth',1.5)
ylabel('Speedup with screening')
grid on
legend({'Speedup' 'Saturation ratio'})

figure(2)
plot(saturated_ratio,screen_speedup)
xlabel('Saturation ratio')
ylabel('Speedup with screening')
grid on

end
