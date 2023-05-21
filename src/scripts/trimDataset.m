%This is the code that was used to create them trimmed dataset
load('data\mnist.mat');
cv = cvpartition(test.labels,'HoldOut',0.2, 'Stratify', true);
idx = cv.test;
training.images = test.images(:,:,~idx);
training.labels = test.labels(~idx);
training.count = sum(~idx);
test.images = test.images(:,:,idx);
test.labels = test.labels(idx);
test.count = sum(idx);
save('data\mnist_10k.mat', 'training', 'test');
