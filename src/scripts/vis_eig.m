om = result.results(1).run.omegaMatrix;
rel = om * om.';
[V, D] = eig(rel);
Vt = V.';
for i=1:hiddenSize
    dec = autoenc.decode(Vt(:,i));
    %im = rescale(dec);
    im=dec;
    subplot(1,hiddenSize,i);
    imshow(im);
end