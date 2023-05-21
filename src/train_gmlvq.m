load('models\autoencoder_01.mat');
load('data\mnist_classes01.mat');

xtraining = cell(1,size(training.images,3));
for i=1:size(training.images, 3)
    xtraining{i}=squeeze(training.images(:,:,i));
end
xencoded = encode(autoenc,xtraining);
xencoded = transpose(xencoded);


lt = LabelTransformer(unique(training.labels));
transformedLabels = lt.transform(training.labels);

gmlvq = GMLVQ.GMLVQ(xencoded, transformedLabels,GMLVQ.Parameters(), 10);

result = gmlvq.runSingle();
result.plot();

nPrototypes = size(result.run.prototypes,1);

prototypes = result.run.prototypes .* repmat(result.run.stdFeatures,nPrototypes,1)...
    + repmat(result.run.meanFeatures, nPrototypes, 1);
prototypes = transpose(prototypes)              ;

origProdotypes = decode(autoenc, prototypes);

subplot(1,2,1);
imshow(origProdotypes{1});
subplot(1,2,2);
imshow(origProdotypes{2});

save("models\gmlvq.mat", "result", "lt", "origProdotypes");
