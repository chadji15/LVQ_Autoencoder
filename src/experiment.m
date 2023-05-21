function experiment(hiddenSize, classes, datasetPercentage, outputFilePath)
    % Load the mnist dataset
    load('data\mnist.mat');
    
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
    xtrain = {1,size(training.images, 3)};
    for i=1:size(training.images, 3)
        xtrain{i}=training.images(:,:,i);
    end
    
    autoenc = trainAutoencoder(xtrain,hiddenSize,...
            'L2WeightRegularization',0.004,...
            'SparsityRegularization',4,...
            'SparsityProportion',0.15,...
            'MaxEpochs',100,...
            'UseGPU', false);
    
    % encode the training data
    xencoded = encode(autoenc,xtrain);
    xencoded = transpose(xencoded);
    
    % convert the labels to the range 1-N
    lt = LabelTransformer(unique(training.labels));
    transformedLabels = lt.transform(training.labels);
    
    % train the gmlvq model
    gmlvq = GMLVQ.GMLVQ(xencoded, transformedLabels,GMLVQ.Parameters(), 10);
    
    result = gmlvq.runValidation(10,10);
    result.plot();
    
    % decode the prototypes
    nPrototypes = size(result.averageRun.prototypes,1);
    
    % revert the zscore transformation that takes place in the toolbox
    prototypes = result.averageRun.prototypes .* repmat(result.averageRun.stdFeatures,nPrototypes,1)...
        + repmat(result.averageRun.meanFeatures, nPrototypes, 1);
    prototypes = transpose(prototypes)              ;
    
    origPrototypes = decode(autoenc, prototypes);
    
    for i = 1:length(classes)
        subplot(1,length(classes),i);
        imshow(origPrototypes{i});
    end

    save(outputFilePath, "autoenc", "result", "origPrototypes", "training", "test", "classes", "hiddenSize");
end