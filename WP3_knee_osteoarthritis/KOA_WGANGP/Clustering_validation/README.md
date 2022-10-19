# Validation of synthetic images using clustering methods

Results of the validation of the synthetic x-ray images using clustering
methods.

## Objectives

 - Analyze the synthetic images by extracting the feature vectors with
 a convolutional neural network trained on ImageNet
 - Cluster the features together with feature vectors extracted from
 real x-ray images to find differences between real and synthetic
 images
 - Cluster the real image features with feature vectors extracted from
 images from OAI-dataset, which was used to train the generative model,
 to find if the differences are intrinsic to the dataset.
 - Visualize the differences in images using class activation mapping

## Methods

### Feature extraction

The features were extracted with VGG19 convolutional neural network trained
on the ImageNet dataset. This resulted in feature vectors with 512 dimensions.

Dimension reduction techniques were used to simplify the extracted features and
to reduce noise.
Two different techniques were compared, PCA and UMAP.
With PCA, 90% of the explained variance was attained at 30 dimensions, at
which point the individual explained variance ratio had fallen under 0.001.
The remaining 10% of the dimensions were assumed to be noise, and different
levels of dimension reduction up to 30 dimensions were used for clustering.

UMAP performed better in conjunction with the clustering methods, but as a
non-linear dimension reduction it produces feature vectors that are difficult
to compare with each other.
Thus UMAP was used to find the cluster labels and PCA projection was used
to explain the results when needed.

### Clustering

Two different clustering methods were used, Kmeans++ and OPTICS. Both of these
were implemented usinf the scikit-learn -library
The feature vectors reduced using PCA were difficult to cluster, as the
features did not form clear clusters, only a few more dense areas with
slightly more sparse areas between.

![Features reduced to 2 dimensions using PCA](\2dpca_real_labels.png
"Features embedded to 2 dimensions using PCA")

Both were successful in clustering the UMAP embedded features. This was due to
UMAP's tendency to push the feature vectors which are similar to each other 
even closer, and separate those which are not even further. This can produce
clusters which are easy label. They cannot be assumed to be spherical or
to have similar variances, but in this case even Kmeans performed well.

### Class Activation Mapping

The clustered images were then visualized with the help of Grad-CAM, a class
activation mapping technique.
Grad-CAM calculates the gradients with regards to the class obtained from
the prediction task in a (convolutional) neural network.
This effectively shows which parts of an image contributed most to the 
classification of the image to a particular class.
These can then be used to visualize these gradients by generating heatmaps
and superimposing them onto the original images.

As there is no clear classes to form the gradients on (the cluster labels
cannot be used), the feature vectors themselves were used to calculate
the gradients. An attempt to use the PCA projection as the final layer
wrt. the gradients are calculated was made, but curiously this did not
have noticeable impact to the heatmaps generated.

## Results 

