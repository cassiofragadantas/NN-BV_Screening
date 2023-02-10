function V = sparseprod(A,S,i,j)

%Calculate entries indicted by index vectors I and J
%from a full matrix product of A and S

V=zeros(length(i),1);

for k=1:length(i)
  V(k) = A(i(k),:)*S(:,j(k));
end

