function [theta, ATtheta] = dualUpdateKL(res,ATres,tdual,Atdual)
% Returns a feasible dual point (theta) and the product A.'*theta (ATtheta)
%
% --Input arguments--
%   res: -df(Ax)/d(Ax) = (y-Ax)/Ax in the KL case
%   ATres: A.'*res = -df/dx
%   normA: (n x 1) vector containing the L2 norms of the columns of A
%   Atdual: (n x 1) vector containing the dot product A.'*tdual
%   t: (n x 1) dual direction vector tdual

% 1) Dual translation (to fall inside dual feasible set)
epsilon = max(ATres./Atdual.'); %max(ATres) also works if A is normalized, but is slightly worse.
theta = res - epsilon*tdual;
% 2) Rescale (to fall inside dual function domain)
scale = max(-min(theta)+1e-6,1);
theta = theta/scale;

ATtheta = (ATres - epsilon*Atdual.')/scale; %= A.'* theta; Should be zero at coordinates xj ~= 0 and negative otherwise