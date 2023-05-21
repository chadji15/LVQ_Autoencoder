load('data\mnist_classes01.mat');
%%
hiddenSize=25;
xtrain = {1,size(training.images, 3)};
for i=1:size(training.images, 3)
    xtrain{i}=training.images(:,:,i);
end
%%
autoenc = trainAutoencoder(xtrain,hiddenSize,...
        'L2WeightRegularization',0.004,...
        'SparsityRegularization',4,...
        'SparsityProportion',0.15,...
        'MaxEpochs',100,...
        'UseGPU', true);
%%
xtest = cell(1, size(test.images,3));
for i=1:size(test.images, 3)
    xtest{i}=squeeze(training.images(:,:,i));
end
xReconstructed = predict(autoenc,xtest);
%%
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(xtest{i});
end
%%
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(xReconstructed{i});
end
%%
save('models\autoencoder_12.mat', 'autoenc');