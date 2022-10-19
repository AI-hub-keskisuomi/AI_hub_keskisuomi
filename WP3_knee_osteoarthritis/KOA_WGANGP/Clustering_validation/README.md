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
Total number of images used was 4000 in each of the sets with 2000 being of
healthy knees (KL 0 and 1) and 2000 of knees with OA (KL 2, 3 and 4). When
comparing two datasets a total of 8000 images were used per experiment.

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

### MOST vs. synthetic images

Using PCA to reduce the data into two dimensions and Kmeans clustering
the optimal number of clusters was found be either 3 or 4, using
inertia values and elbowing method.
When adding more dimensions this value tended towards 4.
OPTICS was not used on the PCA-embedded features as the distribution
of the features was such that a lot of hyperparameter tuning was needed
to produce any meaningful results.

When reducing the dimensions using UMAP, on both clustering methods 4
clear clusters were observed.
OPTICS was chosen as the main method as it had the potential to find
subclusters.
With OPTICS these 4 cluster separated the MOST x-rays from synthetic with over
99% accuracy, with cluster 0 comprising of synthetic images, and MOST x-rays
being divided between clusters 1, 2 and 3.
There was no clear division in KL-grades between the three MOST clusters.
Cluster 1 was 59% of healthy knees (KL 0 and 1), and cluster
3 was 71% of knees with OA (KL 2, 3, and 4). Cluster 2 was smaller with less
than 500 x-rays and was 58% healthy knees.

![Features reduced to 2 dimensions using UMAP](\umap_2d_most_vs_fake.png
"Features embedded to 2 dimensions using UMAP")

![OPTICS clustering in 2 dimensions using UMAP](\umap_2d_optics.png
"OPTICS clustering in 2 dimensions using UMAP")

### MOST vs. OAI

The results were validated using OAI images in place of the synthetic x-rays.
The experiments conducted were exactly the same as before; the features
were extracted from both datasets, dimension reduction tehniques were used
on the combined dataset, and the resultant embedded feature vectors were
clustered.

The clustering results were similar to the previous clustering with synthetic
images. The Adjusted Rand Index for these two clusterings were over 0.94 in
in every dimension up to 30. As majority of the information in these lower
30 dimensions (rest are assumed to be more or less noise), they were not
explored.

This indicates that the synthetic x-rays are a good representation of the
real OAI images, the dataset which was used to train the generative model,
and are indistiguishable from each other.
However, the differences between the datasets OAI and MOST are still
very large. While the generative model is good at generating images
indistinguishable from OAI images, it is still limited at generating
"OAI images".

### Grad-CAM visualizations

To visualize concretely the images clustered in the MOST/synthetic-clustering
the features were reduced to 30 dimensions using UMAP and then clustered
using OPTICS with stricter hyperparameters.
This yielded 23 subclusters with most of the feature vectors discarded as
noise.

![30 dim OPTICS subclusters, visualized in 2 dim](\umap_30d_subclusters.png
"30 dim OPTICS subclusters, visualized in 2 dim")


The centermost feature vectors of these clusters (using euclidean distance)
was chosen as the cluster representative.
The images corresponding to these feature vectors and their heatmaps can
be seen in the image below.

![Cluster representatives and Grad-CAM heatmaps](\readme_gradcam.png
"Cluster representatives and Grad-CAM heatmaps")

The Grad-CAM results show no clear differences between the clusters. The clusters 0-7
were mostly synthetic images and the Grad-CAM visualizations show that the VGG19 -
network focused on sharp edges, and put emphasis on the joint space an tibial spike, while
also focusing on the femur and patella. The clusters 8-14 were real images and while clearly
distinct from the synthetic images in the UMAP projection, they were the closest when using
PCA. The Grad-CAM heatmaps could also indicate this, although the difference is not clear.

Subclusters 15-18 were part of the smallest of the clusters. A noticeable "striping" could
be seen in all of the images in the clusters. The heatmaps show that the network was not
totally distracted by this features, but as they were distinctly clustered together, this might
have been the most prominent feature.

Subclusters 19-23 were a part of the final larger cluster. This cluster consisted of the blurrier
image from MOST. The heatmaps seem to light smaller areas, and this could possibly be due
to the network struggling to find distinct features.