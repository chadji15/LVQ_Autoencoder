function custom_autoenc_experiment(hiddenSize, classes, datasetPercentage, ...
    doztr, outputFilePath)
    % Load the mnist dataset
    load('data/mnist.mat');
    
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
    lt = LabelTransformer(unique(training.labels));
    transformedLabels = lt.transform(training.labels);
    
    % train the gmlvq model
    gmlvq = GMLVQ.GMLVQ(xencodedt, transformedLabels,GMLVQ.Parameters("doztr", doztr), 30);
    
    result = gmlvq.runValidation(10,10);
    
    % decode the prototypes
    nPrototypes = size(result.averageRun.prototypes,1);
    prototypes = result.averageRun.prototypes;

    if doztr
        % revert the zscore transfor mation that takes place in the toolbox
        prototypes = result.averageRun.prototypes .* repmat(result.averageRun.stdFeatures,nPrototypes,1)...
            + repmat(result.averageRun.meanFeatures, nPrototypes, 1);
        
    end
    
    prototypes = transpose(prototypes)              ;
        
    origPrototypes = autoenc.decode(prototypes);
    for i = 1:length(classes)
        subplot(1,length(classes),i);
        imshow(squeeze(origPrototypes(:,:,:,i)), []);
    end

    save(outputFilePath, "autoenc", "result", "origPrototypes", "training", "test", "classes", "hiddenSize");
    result.plot();
end