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

invidx = ~idx;
idx = idx(1:1000);
invidx = invidx(1:1000);


scatter(enc_dist(invidx),dist(invidx));
hold;
scatter(enc_dist(idx),dist(idx));
hold;
coef = polyfit(enc_dist, dist,1);
h = refline(coef(1), coef(2));
h.Color = 'g';
h.LineStyle = "--";
h.LineWidth = 2;

xlabel('Encoded Distance')
ylabel('Decoded Distance')

hold;
coef2 = polyfit(enc_dist, dist,2);
x = linspace(min(enc_dist), max(enc_dist));
p = plot(x, polyval(coef2,x), "--magenta");
p.LineWidth = 2;

pred_dist = polyval(coef,enc_dist); 
pred_dist2 = polyval(coef2,enc_dist); 

disp("Linear model RMSE") 
rmse1 = rmse(pred_dist, dist)
disp("Parabolic model RMSE") 
rmse2 = rmse(pred_dist2, dist)
    