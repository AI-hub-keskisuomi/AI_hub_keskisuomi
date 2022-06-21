# Knee Osteoarthitis Wasserstein Generative Adversarial Networks with Gradient Penalty

Provided training/generation scrips for generating synthetic x-ray images with WGANGP

### Problem Objectives

- Synthetize x-rays of various osteoarthitis severities. 
- Substitute real data with synthetic in classification task (anonymize)
- Augment existing data in classification task
- Explore laten variables

Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![](image27.gif)  |  ![](https://...Ocean.png)


>Synthetic X-Rays interpolation in latent variable space.

### Running the material

- In order to generate images first load the architecture by running WGANGP_KOA.py;
- Proceed to load the weights (KL01 or KL234) file by specifying location with:

```python
wgan.load_weights ( 'path_to_weights')
```
- Alternatevely, in order to train or re-train (begin with loaded weights) the model you need to pass (after pointing to the data folder etc):

```python
wgan.fit ( dataset , batch_size , epochs )
```
- To generate the images from the pre-trained model you need run the Generate_KL.py and specify the batches and how many images within each batch (GPU RAM limitation)
- The model was trained with OAI images, resolution supported is exactly 210 x 210 pixels.
- Example folder contains synthetic images for KL01 and KL234 osteoarthitis severities.
