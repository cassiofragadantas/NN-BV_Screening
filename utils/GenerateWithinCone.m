function pts = GenerateWithinCone(ax,r,nb_pts,border)
% Inputs: ax (cone axis), r (controls cone opening angle), 
%         nb_pts (number of points to be generated), border (samples only on cone border if true)
if nargin < 4, border = false; end

n=length(ax);  % dimension
ax=normc(ax);  % cone axis

% Get basis for the hyperplane orthogonal to ax
B=null(ax');  % Q=[ax,B] is orthogonal

% Generate points in a r-ball of the hyperplane
if border
    coord=normc(randn(n-1,nb_pts))*r;
else 
    coord=normc(randn(n-1,nb_pts)).*nthroot(rand(1,nb_pts),n-1)*r;
end
pts=ax+B*coord;
pts=normc(pts);

% Illustration (to be uncommented)
% if n==2
% figure;
% plot(pts(1,:),pts(2,:),'x');  axis equal; grid;axis([0 1 0 1]);
% hold all; plot([0 ax(1)],[0 ax(2)]);
% end
% 
% if n==3
% figure;
% scatter3(pts(1,:),pts(2,:),pts(3,:)); axis([0 1 0 1]);
% hold all; plot3([0 ax(1)],[0 ax(2)],[0 ax(3)]);axis equal;
% end