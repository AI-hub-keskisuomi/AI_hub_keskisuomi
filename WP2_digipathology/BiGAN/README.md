# BiGAN
BiGAN code (Timo Ojala)

==================================

## Introduction

BiGAN is a generative model based on the concept of Generative Adversarial Networks (GANs). What BiGAN adds to the basic GAN concept is that in addition to Generator network (G) that tries to turn a random vector ("latent" or "latent vector") into an image indistinguishable (by a Discriminator network (D)) from examples of the used dataset, there's also an Encoder network (E), that reverses the Generator's process by turning examples from the dataset into latent vectors indistinguishable (by D) from the randomly generated latents. An additional important detail is that these images (generated and examples from dataset) and latents (randomly generated and encoded from dataset examples) are fed into the Discriminator network as pairs, so the discriminator can learn to match the latent vectors to images. A motivation for using BiGAN networks is that the Encoder can later be used as a feature extractor for images.

While BiGAN can be seen as somewhat similar to an Auto-encoder, it is important to remember that BiGAN is not trained to be an auto-encoder. BiGAN Encoder E is trained through Discriminator network D that tries to discriminate between real images with encoded latents and fake images with random latent. For semi-supervised learning, various unsupervised learning techniques can be tried to see which ones produce desirable feature extractors.

An improved model for BiGAN concept called BigBiGAN also exists, implementing this could be an useful way to improve performance if continuing development of this code in future. BigBiGAN Discriminator includes sub-networks for discriminating between images and latents only. The scores of these sub-networks are used in addition to the score of the joint discriminator.

Articles:

(BiGAN)

Jeff Donahue, Philipp Krähenbühl & Trevor Darrell, (2017) Adversarial Feature Learning https://arxiv.org/abs/1605.09782

(BigBiGAN)

Jeff Donahue & Karen Simonyan, (2019) Large Scale Adversarial Representation Learning https://arxiv.org/abs/1907.02544

==================================

## Experimental functions and layers

This implementation includes a lot experimental tricks, layers and functions that mostly try to fix difficulties in GAN training or failures of the model in some specific areas, some of them are commented out or not used because they do not improve the results at the time being. It also implements many published improvements of GAN training that are newer than the original BiGAN paper.


Some work has been done to implement experimental activation functions and pooling layers. The activation functions are mostly intended to be used at the final layers of G, E or D. Encoder output should roughly match the latent distribution, which at the time being is normal distribution (centered at 0 and with std of 1), for which linear activation (= no activation) seems to be the only feasible default function. New pooling layers were implemented due to uncertainty on whether either max or average pooling were good choices for Discriminator. The one at use in models at the time of writing is a mixed pooling functiton, where the pooling layer learns to mix both max pooling and average pooling for each channel.

==================================

## Mode Collapse

Mode collapse in GAN training mostly means a failure case where Generator keeps generating similar images and moving on to new similar images when Discriminator catches on.

Implementing Unrolled GAN training might solve mode collapse in training if it happens. This implementation might require some low level Tensorflow functionality like the graph editor. Another potential method for combating mode collapse could be Spectral Regularization.

While training the model, mode collapse seems to be more prevalent in Encoder side of things, to help with this, the G and E training step includes a somewhat improvised form of regularization that penalizes standard distribution/variance of encoded vectors getting far from the expected values. This regularization looks like the following:
    #MSE between vectors
    mse = tf.reduce_mean(tf.square(tf.expand_dims(encoded_batch, axis=0) - tf.expand_dims(encoded_batch, axis=1)), axis=[1,2])
    encoded_sample_similarity = -1.5 * tf.math.tanh(mse)
    variance_loss = variance_loss_multiplier * tf.maximum(tf.losses.MSE(1.0, tf.math.reduce_variance(encoded_batch, axis=-1)) - variance_loss_tolerance, 0.0)
This regularization does not necessarily provide working gradients that push the model directly towards optimum weights, but it seems to help break collapsing Encoder into providing better distributed gradients again. For that reason, the weight of this regularization during training varies in training loop depending on how close the encodings are to each other. From quick searching over the internet, the closest method to this seems to be Mode Regularized Generative Adversarial Networks.

Articles on the methods:

Unrolled GANs:

    Luke Metz, Ben Poole, David Pfau & Jascha Sohl-Dickstein, (2017) Unrolled Generative Adversarial Networks https://arxiv.org/abs/1611.02163
    
Spectral Regularization:

    Kanglin Liu1, Wenming Tang1, Fei Zhou1 & Guoping Qiu, (2019) Spectral Regularization for Combating Mode Collapse in GANs https://arxiv.org/abs/1908.10999
    
Mode Regularized Generative Adversarial Networks

    Tong Che, Yanran Li, Athul Paul Jacob, Yoshua Bengio & Wenjie Li, (2017) Mode Regularized Generative Adversarial Networks https://arxiv.org/abs/1612.02136

==================================

## Applying to new datatypes and evaluation

Performance of BiGAN can be evaluated (when there's no supervised task available) using the auto-encoder task, which is not exactly the task of a BiGAN, but will roughly correspond to Generator performance and Encoder's ability to invert the Generator, and also likely tell the model's ability to perform desired future tasks. A problem with MNIST GANs is often that the number shapes tend to blend together in the dataset itself, so unsupervised models might not easily learn the distinctions that might seem obvious to humans at first. 

The architecture of G, E and D need to be modified if the model is to be applied for new datasets. MNIST has been primarily used for development to diagnose problems with the framework, seeing as it should be a relatively light and simple dataset to train on.

For training on histology images, I have tried using U-net style of architecture for a few reasons. U-net is often used with larger images as a type of fully convolutional architecture, where the resolution of the image is reduced in the first half but increased back to the original size in second half. This allows processing larger images without having to reduce the size of output. Being a fully convolutional architecture, U-net might suit data that is spatially more independent (like histology data usually seems to be) than data like faces or animals or anything that has a larger, single object of interest. Fully convolutional architectures can often be thought of as processing the input in a sliding window manner, but in parallel. 

==================================

## Datasets in file

The near start of the BiGAN.py file is included (and mostly commented out) 5 datasets for training the BiGAN. These datasets are not necessarily selected as best examples for real world application, but more for easy examples for developing the base functionality. PCam dataset is a simplified and small version of CAMELYON16 dataset, the data being smaller image patches. PCam would be a good dataset choice for easy histopathology dataset.

PCam dataset needs to be downloaded separately and the path in code needs to be changed to the file/folder. Full PCam dataset will probably need around 9 GB of RAM when loaded onto memory at once, otherwise an iterator that loads the data to memory in smaller parts at a time needs to be implemented.

Datasets in total:
    -MNIST
    -CIFAR-10
    -32x32 Street View House Numbers
    -PCam
    -(Scaled down) Celeb-A
    
The current implementation of Celeb-A requires a fairly long preprocessing time and over 8 GB of memory to use. Preprocessing has to be done only once, after that the dataset is saved in a numpy file that can be loaded. For a less bothersome use of this dataset, an iterator that loads the data from hard-disk as needed would be better, the Tensorflow Datasets (tfds) iterator should work with appropriate batch size, though the interface of it will be different and require changes in main training loop plot functions.

Because CNNs can scale poorly for larger image sizes without scaling down on first layers, these datasets are mostly on the smaller size. For applying the BiGAN model on larger image patches, it's important to develop architectures with memory usage in mind. A lot of datasets in tensorflow-datasets package would be easier to use with an iterator made following the instructions in the package documentation, but the current plotting functions at the training loop need a subscriptable iterator. This could be changed to use (for example) some fixed set of images that are reconstructed.

==================================

## Generator head concepts. (This exists in the BiGAN.py file as a comment)

Mostly old network parts/ideas for shaping the latent to something that resembles an image. This can be an important piece of the G network, because whether the image content is independent of across the image or whether it forms a singular coherent shape could be decided here. Due to lack of time to develop, test and to make this into a proper function that returns various forms of these, I will leave these commented out. This part of the generator architecture might be important for future development for various types of data. The layers created here should be moved to the Generator construction function (create_generator) to be used in a Generator architecture.
 
Additional notes:
    The way latent forms the tensor in either dense or convolutional layers also affects how the encoding produced by E relates to the image. Because Dense layer neurons take whole input vectors and produce weighted sums of them (then activation), dense layers are very likely to spread the effects of changes in single latent vector elements across the whole input image.
    A dense layer (or something like an MLP) for G head allows G to better build a big picture, for example faces, characters/digits or various natural images. These do not enjoy full benefits of convolutions, which is, to some degree, the point of using dense layers.
    A fully convolutional architecture for both G head and encoder will make the processing more independent between image regions, but also give the general benefits of convolutions:
        (translational invariance, re-use of feature maps across image in spatial dimensions, reduced number of trained parameters)
    Histological microscopy images tend to be very independent in nature when it comes to smaller scale features (nuclei and such), while there are also larger scale structural features. A fully convolutional G head might be more applicable to these images, but if the images fed to the BiGAN are already relatively small in size, it is not necessarily out of the question to use dense layer G head for them too.
    When considering the options between these two, one could just compare how each performs in practise, but the main question is what kind of fully convolutional G head should one use? The answer to this might regrettably not be obvious, and it could be a major question when designing the architecture.
    Generator head type may introduce additional requirements for latent size and the architecture of Encoder, mainly the final layers after flattening. If changing latent shape to 2D, this will also necessitate changes in (at the very least) training step and printing code, so it is not an endeavor that should be taken lightly.     

WARNING: Working combinations of the code lines might not be grouped together coherently, and some of the parts might not work, either in terms of the network not compiling, or in terms of learning. So, if reading through these and something doesn't make sense, it might be some leftover pieces of code or failed attempts.

==================================

## Architectures

The default architectures are developed on MNIST data, though they could be applied to different tasks aswell. Potential choices for architecture design might be virtually unlimited, but for a more straight forward approach, it might be more important to approach this more as a task of balancing the depths of the network/widths of network layers to improve performance of parts that perform poorly. It would be ideal to also take into consideration how G/E forms the images/encodings from latent/data, and how the effects from each of the latent elements spread across the image spatially.

In order to reduce edge artefacts, the included architectures pad the Generator tensor before convolutional layers and later crop the tensor to correct image size.

==================================

## Gradient Penalty

Gradient Penalty was implemented mostly by mimicking an existing Keras tutorial implementation of it. It is uncertain if the current implementation is perfect. It is modified to fit BiGAN, many BiGAN implementations of it did not seem to work in the setup of my BiGAN implementation. To me, it seemed like many existing GP implementations did not work in current versions of TF/Keras, or required a different way of building the model.

==================================

## Saving the model

Save and load functions can be set to save only weights instead of the full model.
Saving only the weights makes the loading process much more simple, without much risk of something unexpected happening. When saving or loading weights, it's important that the model structure has not been changed between the saved model and the one it's being loaded into. Easy way to checkpoint the code version would be to just copy the current file in with the saved weights since it's just a (relatively) small file as of yet.
Saving and loading the full models with configuration, architecture etc. for G, E and D should work for BiGAN training the way it has been implemented here. It's important to link the models, since the full BiGAN model needs various models to be joined together (so remember this if implementing own load functions elsewhere!).

An example of how to load models would be something like
    gan = BiGAN(latent_size=latent_size, batch_size=batch_size, channels=channels)
    gan.load(
        G_path="2021-06-16 05_24 G", E_path="2021-06-16 05_24 E",
        D_path="2021-06-16 05_24 D", load_as_weights=False)
where the paths for each model folder are defined. When saving the model instead of just weights, the Tensorflow save function creates folders with data inside, the load path should point to the folder.

It might be somewhat tedious having to define path for each model separately, though this way you can mix different parts with load function (assuming the shapes fit together). The load function could be changed to allow loading partial model, so for example the user could load just weights or model for E.

==================================
