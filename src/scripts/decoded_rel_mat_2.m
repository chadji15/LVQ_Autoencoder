rel = result.averageRun.lambda;
Z = diag(result.averageRun.stdFeatures);
rel_inv = Z.' * rel * Z;

[V, D] = eig(rel_inv, 'vector');

[m,idx] = max(D);
primEig = V(:,idx);
primEigIm = autoenc.decode(primEig);

u = reshape(primEigIm,[],1);
rel_dec = u * u.';

data = autoenc.encode(training.images);
data = reshape(data, autoenc.hiddenSize, []);
data = data.';

prots = result.averageRun.prototypes;
prot = prots(1,:);
%invert z score
prot = prot .* result.averageRun.stdFeatures + result.averageRun.meanFeatures;

diff = data - prot;
enc_dist = dot(diff*rel_inv,diff,2);

orig_prot = reshape(origPrototypes(:,:,:,1),[],1);

orig_data = zeros(length(training.images),28*28);
for i=1:length(training.images)
    orig_data(i,:) = reshape(training.images(:,:,i),28*28,1);
end

diff= orig_data - orig_prot.';
dist = dot(diff*rel_dec,diff,2);

lbl = training.labels(1);
idx = training.labels == lbl;

scatter(enc_dist(~idx),dist(~idx));
hold;
scatter(enc_dist(idx),dist(idx));
hold;
coef = polyfit(enc_dist, dist,1);
h = refline(coef(1), coef(2));
h.Color = 'r';


    