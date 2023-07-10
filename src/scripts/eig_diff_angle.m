protdiff = squeeze(origPrototypes(:,:,1,1) - origPrototypes(:,:,1,2));
protdiff2 = squeeze(origPrototypes(:,:,1,2) - origPrototypes(:,:,1,1));

rel = result.averageRun.lambda;
Z = diag(result.averageRun.stdFeatures);
rel_inv = Z.' * rel * Z;
[V, D] = eig(rel_inv, 'vector');

[m,idx] = max(D);
primEig = V(:,idx);
primEigIm = autoenc.decode(primEig);

u = reshape(primEigIm, [], 1);
v1 = reshape(protdiff, [],1);
v2 = reshape(protdiff2, [], 1);

cossim1 = dot( v1, u)/(norm(v1)*norm(u))
cossim2 = dot(v2, u)/(norm(v2)*norm(u))
