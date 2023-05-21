%This is the code that was used to create them trimmed dataset
classes = [0 1];

load('data\mnist.mat');

idx = ismember(training.labels, classes);
training.images = training.images(:,:,idx);
training.labels = training.labels(idx);
training.count = sum(idx);

idx = ismember(test.labels, classes);
test.images = test.images(:,:,idx);
test.labels = test.labels(idx);
test.count = sum(idx);

save('data\mnist_classes12.mat', 'training', 'test');
