rel = result.averageRun.lambda;
Z = diag(result.averageRun.stdFeatures);
rel_inv = Z.' * rel * Z;
[V, D] = eig(rel_inv);

for i=1:hiddenSize
    dec = autoenc.decode(V(:,i));
    %im = rescale(dec);
    im=dec;
    subplot(1,hiddenSize,i);
    imshow(im, []);
end