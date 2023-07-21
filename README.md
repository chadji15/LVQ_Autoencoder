# Autoencoder-based Interpretable Classification using LVQ and Relevance Learning

In this project we will investigate the combination of autoencoder networks  with Learning Vector Quantization and Relevance Learning in the context of classification problems. We will consider a novel set-up for interpretable classification of high-dimensional data: 

In a first training phase an autoencoder is trained in an unsupervised manner, yielding a faithful low-dimensional representation of the data in its bottleneck layer (latent space).

Next, a prototype-based classifier (LVQ), potentially equipped with relevance learning as in GMLVQ , will be trained in the low-dim. space to perform the desired classification. Eventually, the decoder obtained in the first phase can be used to (nonlinearly) project back the prototypes to the original high-dim.\ feature space. This facilitates better interpretation of the classifier with the domain expert and allows for visual inspection, for instance in case of image classification tasks.

## Setup
---

For the experiments, Matlab 2023a was used.

Required Toolboxes:
- Deep Learning Toolbox
- Parallel Processing Toolbox
- No-nonsense GMLVQ (https://www.cs.rug.nl/~biehl/gmlvq)
  
The GMLVQ toolbox needs to be added to the Matlab path. Further instruction on that can be found in the provided link.

Dataset: MNIST dataset for matlab: https://lucidar.me/en/matlab/load-mnist-database-of-handwritten-digits-in-matlab/. Put it under the "data" directory (data/mnist.mat).

## Execution:
---
Most functionality lives inside the *custom_autoencoder_experiment.m* function. It trains an autoencoder on the specified subset of the data, then trains the GMLVQ algorithm and finally produces some plots and saves the workspace.

Arguments:

1. hiddenSize: the size of the bottleneck layer of the autoencoder
2. classes: an array with digits. Specifies the subset of the data to use. Example: [0 1].
3. datasetPercentage: float in the range (0,1]. Usually 1, but if your machine runs out of memory for bigger experiments, specify a lower number to "trim" the dataset.
4. doztr: true/false. Whether to do a z-score transformation to the data before applying GMLVQ.
5. outputFilePath: the path where the function will save the workspace.

The following scripts can be executed after a workspace saved with the above function is loaded:
- decoded_rel_mat_2.m: calculates the relevance matrix in the decoded space and the encoded and decoded distances from one of the prototypes to all the datapoints. Fits a linear and parabolic model to these datapairs and prints the RMSE for both. Plots all of the above in a scatterplot.
- vis_eig.m: displays the decoded principal eigenvector as an image.
- vis_prot_decoded_diff.m: displays the difference between the two prototypes after they have been decoded.
- eig_diff_angle: calculates the cosine similarity between the decoded principal eigenvector and the prototype difference.

Also provided is the visu_3d.m function that takes a GMLQ.Result object and projects the dataset and prototypes on the 3 principal eigenvectors of the relevance matrix.