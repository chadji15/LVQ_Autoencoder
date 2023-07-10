doztr = true;

rel = result.averageRun.lambda;
Z = diag(result.averageRun.stdFeatures);
rel_inv = Z.' * rel * Z;
if doztr
    rel = rel_inv;
end

[V, D] = eig(rel_inv, 'vector');

[m,idx] = max(D);
primEig = V(:,idx);
primEigIm = autoenc.decode(primEig);

u = reshape(primEigIm,[],1);
rel_dec = u * u.';

data = result.averageRun.trainingData.featureVectors;

%invert z score
if doztr
data = data .* repmat(result.averageRun.stdFeatures,length(data),1)...
            + repmat(result.averageRun.meanFeatures, length(data), 1);
end

prots = result.averageRun.prototypes;
prot = prots(1,:);
%invert z score
if doztr
    prot = prot .* result.averageRun.stdFeatures + result.averageRun.meanFeatures;
end

diff = data - prot;
enc_dist = dot(diff*rel,diff,2);

orig_prot = reshape(origPrototypes(:,:,:,1),[],1);

orig_data = zeros(size(data,1),28*28);
for i=1:size(data,1)
    orig_data(i,:) = reshape(training.images(:,:,i),28*28,1);
end

diff= orig_data - orig_prot.';
dist = dot(diff*rel_dec,diff,2);

scatter(enc_dist,dist);
coef = polyfit(enc_dist, dist,1);
hold;
h = refline(coef(1), coef(2));
h.Color = 'r';


    