function multiclass_binary_experiment(hiddenSize, group1, group2, prototypesPerGroup, ...
    datasetPercentage, outputFilePath)
    % Load the mnist dataset
    load('data/mnist.mat');

    classes = [group1 group2];
    
    % Keep only the labels that interest us
    idx = ismember(training.labels, classes);
    training.images = training.images(:,:,idx);
    training.labels = training.labels(idx);
    training.count = sum(idx);
    
    idx = ismember(test.labels, classes);
    test.images = test.images(:,:,idx);
    test.labels = test.labels(idx);
    test.count = sum(idx);
    
    % Trim the dataset if needed
    if datasetPercentage < 1
        cv = cvpartition(training.labels,'HoldOut',datasetPercentage, 'Stratify', true);
        idx = cv.test;
        training.images = training.images(:,:,~idx);
        training.labels = training.labels(~idx);
        training.count = sum(~idx);
    
        cv = cvpartition(test.labels,'HoldOut',datasetPercentage, 'Stratify', true);
        idx = cv.test;
        test.images = test.images(:,:,~idx);
        test.labels = test.labels(~idx);
        test.count = sum(~idx);
    end
    
    % Train the autoencoder
    % xtrain = {1,size(training.images, 3)};
    % for i=1:size(training.images, 3)
    %     xtrain{i}=training.images(:,:,i);
    % end

    autoenc = CustomAutoencoder(hiddenSize, training.images);
    
    % encode the training data
    xencoded = autoenc.encode(training.images);
    xencodedr = reshape(xencoded, hiddenSize, size(training.images,3));
    xencodedt = transpose(xencodedr);
    
    % convert the labels to the range 1-N
    transformedLabels = training.labels(:);
    idx = ismember(transformedLabels, group1);
    transformedLabels(idx) = 1;
    transformedLabels(~idx) = 2;

    % train the gmlvq model
    gmlvq = GMLVQ.GMLVQ(xencodedt, transformedLabels,GMLVQ.Parameters(), 30, ...
        [ones(1,prototypesPerGroup) ones(1,prototypesPerGroup)*2]);
    
    result = gmlvq.runValidation(10,10);
    
    % decode the prototypes
    nPrototypes = size(result.averageRun.prototypes,1);
    
    % revert the zscore transformation that takes place in the toolbox
    prototypes = result.averageRun.prototypes .* repmat(result.averageRun.stdFeatures,nPrototypes,1)...
        + repmat(result.averageRun.meanFeatures, nPrototypes, 1);
    prototypes = transpose(prototypes)              ;
    
    origPrototypes = autoenc.decode(prototypes);
    
    for i = 1:size(origPrototypes,4)
        subplot(1,size(origPrototypes,4),i);
        imshow(squeeze(origPrototypes(:,:,:,i)));
    end

    save(outputFilePath, "autoenc", "result", "origPrototypes", "training", "test", "classes", "hiddenSize");
    result.plot();
end