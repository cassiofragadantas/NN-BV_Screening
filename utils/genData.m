function [A,y,n,tdual] = genData(m,n,density_x,exp_type,noise_type,noise_val,normalizeA)
    % Default input values
    if nargin < 2, error('Please provide at least problem dimensions (n,m)'); end    
    if nargin < 3, density_x = 0.05; end
    if nargin < 4, exp_type = 'synthetic'; end
    if nargin < 5, noise_type = 'gaussian_std'; end
    if nargin < 6 
        if strcmp(noise_type,'gaussian_std')
            noise_val = 0.1; % noise standard deviation
        elseif strcmp(noise_type,'gaussian_snr')
            noise_val = 10;  % SNR in dB
        end
    end
    if nargin < 7, normalizeA = true; end

    % Generate A
    switch exp_type
        case {'tall', 'case1'} 
            % Tall A (m>=n) with random unit-norm columns
            if n > m, warning('Matrix A should be tall, setting n = m/2.'); n=floor(m/2); end
            A = randn(m,n); 
        case {'orthogonal', 'case 2'} 
            % Random orthogonal (square) A
            if n ~= m, warning('Matrix A should be square, setting n = m.'); n=m; end
            A = randn(m,n); A = orth(A);
        case {'non-negative', 'case3', 'synthetic'} 
            % Non-negative A with random unit-norm columns
            A = abs(randn(m,n));             
        case {'columnCorr', 'case4'} 
            % One (the first) column is positively correlated to all others
            % Create a big number of columns and keep only the ones that 
            % are positively correlated to the first column
            A = randn(m,1);
            while size(A,2) < n
                newCols = randn(m,2*n);
                correl = A(:,1).'*newCols;
                A = [A newCols(:,correl>0)];
            end
            A = A(:,1:n);            
        case 'conv' 
            % Convolution matrix
            len = 20; % filter size ~6*len + 1
            if n ~= m, warning('Matrix A should be square, setting n = m.'); n=m; end
            normalizeA = false;  % normalizeA is forced to false
            h = fspecial('gaussian',[1,2*ceil(3*len)+1],len); % 1D filter
            halfh = (length(h)-1)/2;
            A = toeplitz([h zeros(1,n-halfh-1)], [h(1) zeros(1,n+halfh-1)]);
            A = A(halfh+1:end,1:end-halfh);            
        case 'cond' 
            % Matrix with controlled condition number
            A = abs(randn(m,n)); % random A with unit-norm columns
            if normalizeA, A = A./sqrt(sum(A.^2)); end
            [U, S, V] = svd(A,'econ');
            cond_factor = 1000; % try 10^-3, 10^-1, 1, 10, 1000
            A = U*diag((diag(S)-S(end))*cond_factor + S(end))*V.';
        otherwise
            load_dataset
    end
    if normalizeA, A = A./sqrt(sum(A.^2)); end

    % Generate y
    %     y = abs(randn(m,1)); % Fully random y
    x_golden = sprand(n,1,density_x); 
    y = A*x_golden;
    y = y/norm(y); 
    
    %Add noise
    if strcmp(noise_type,'poisson')
        y = poissrnd(full(y));
    elseif strcmp(noise_type,'gaussian_std') % gaussian with std
        y = y + noise_val*randn(size(y));
    elseif strcmp(noise_type,'gaussian_snr') % gaussin with snr
        y = y + (10^(-noise_val/20)*norm(y)/sqrt(m))*randn(m,1);
    end

    if any(strcmp(exp_type,{'synthetic','case3'})), y = max(y,0); end   
    
    % Choice of dual translation direction (-t vector on the paper)
    switch exp_type
        case {'tall', 'case1'}
            % Solution of the system A.'*z=b for any positive b
            b = abs(randn(n,1));
            z = A.'\b;
        case {'orthogonal', 'case2'}
            % Arbitrary linear combination of the columns of A
            z = A * rand(n,1);
        case {'columnCorr', 'case4'}
            z = A(:,1);
        otherwise % Non-negative A
            assert(all(A(:)>=0),'Dual direction not implemented for this type of matrix A.')
            z = ones(m,1);

            % Other possibilities:
            % Average of the columns
%             z = sum(A,2); z = z./norm(z);
            % Controlled correlation w.r.t. all-ones vector
%             z = ones(m,1); z(1:floor(m/1.04)) = 0.001; z = z/norm(z); % m/1.1 -> correl 0.3, 1.04 -> 0.2
%             %z.'*ones(m,1)/(sqrt(m)*norm(z)) %Print correlation
    end
    tdual = z./norm(z);
    assert(all(A.'*tdual>0),'Vector tdual has to be positively correlated with all columns of A.');

end