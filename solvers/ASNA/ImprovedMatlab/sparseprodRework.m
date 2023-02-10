%Calculate entries indexed by index vectors i and j
%from a full matrix product of A and S
function V = sparseprodRework(A,S,i,j)

V=zeros(length(i),1);

for k=1:length(i)
  V(k) = sum(A(:,i(k)).*S(:,j(k)));
end

