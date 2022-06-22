#based on https://keras.io/examples/generative/wgan_gp/

import glob
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

IMG_SHAPE = (210 , 210 , 1)
noise_dim = 50

filelist = glob.glob ( r'W:\Deepkoagans\FID\Reals\Reals\*.png' )
x_0 = np.array ( [ np.array ( Image.open ( fname ).convert ( 'L' ) ) for fname in filelist ] )
x_0 = x_0.astype ( "float32" ) / 255
x_0 = np.reshape ( x_0 , (-1 , 210 , 210 , 1) )


def conv_block (
        x ,
        filters ,
        activation ,
        kernel_size=(3 , 3) ,
        strides=(1 , 1) ,
        padding="same" ,
        use_bias=True ,
        use_bn=False ,
        use_dropout=False ,
        drop_value=0.5 , ):
    x = layers.Conv2D ( filters , kernel_size , strides=strides , padding=padding , use_bias=use_bias ) ( x )

    if use_bn:
        x = layers.BatchNormalization ( ) ( x )
    x = activation ( x )

    if use_dropout:
        x = layers.Dropout ( drop_value ) ( x )
    return x


def get_discriminator_model ():
    img_input = layers.Input ( shape=IMG_SHAPE )
    x = layers.ZeroPadding2D ( (2 , 2) ) ( img_input )
    x = conv_block (
        x ,
        64 ,
        kernel_size=(5 , 5) ,
        strides=(2 , 2) ,
        use_bn=False ,
        use_bias=True ,
        activation=layers.ELU ( 0.2 ) ,
        use_dropout=False ,
        drop_value=0.3 ,
    )
    x = conv_block (
        x ,
        128 ,
        kernel_size=(5 , 5) ,
        strides=(2 , 2) ,
        use_bn=False ,
        activation=layers.ELU ( 0.2 ) ,
        use_bias=True ,
        use_dropout=True ,
        drop_value=0.3 ,
    )
    x = conv_block (
        x ,
        256 ,
        kernel_size=(5 , 5) ,
        strides=(2 , 2) ,
        use_bn=False ,
        activation=layers.ELU ( 0.2 ) ,
        use_bias=True ,
        use_dropout=True ,
        drop_value=0.3 ,
    )
    x = conv_block (
        x ,
        312 ,
        kernel_size=(5 , 5) ,
        strides=(2 , 2) ,
        use_bn=False ,
        activation=layers.ELU ( 0.2 ) ,
        use_bias=True ,
        use_dropout=False ,
        drop_value=0.3 ,
    )

    x = conv_block (
        x ,
        422 ,
        kernel_size=(5 , 5) ,
        strides=(2 , 2) ,
        use_bn=False ,
        activation=layers.ELU ( 0.2 ) ,
        use_bias=True ,
        use_dropout=False ,
        drop_value=0.3 ,
    )
    x = layers.Flatten ( ) ( x )
    x = layers.Dropout ( 0.2 ) ( x )
    x = layers.Dense ( 1 ) ( x )

    d_model = keras.models.Model ( img_input , x , name="discriminator" )
    return d_model


d_model = get_discriminator_model ( )
d_model.summary ( )


def upsample_block (
        x ,
        filters ,
        activation ,
        kernel_size=(3 , 3) ,
        strides=(1 , 1) ,
        up_size=(2 , 2) ,
        padding="same" ,
        use_bn=False ,
        use_bias=True ,
        use_dropout=False ,
        drop_value=0.3 ,
): #sad parenthesis
    x = layers.UpSampling2D ( up_size ) ( x )
    x = layers.Conv2D (
        filters , kernel_size , strides=strides , padding=padding , use_bias=use_bias
    ) ( x )

    if use_bn:
        x = layers.BatchNormalization ( ) ( x )
    if activation:
        x = activation ( x )
    if use_dropout:
        x = layers.Dropout ( drop_value ) ( x )
    return x


def get_generator_model ():
    noise = layers.Input ( shape=(noise_dim ,) )
    x = layers.Dense ( 14 * 14 * 512 , use_bias=False ) ( noise )
    x = layers.BatchNormalization ( ) ( x )
    x = layers.ELU ( 0.2 ) ( x )

    x = layers.Reshape ( (14 , 14 , 512) ) ( x )
    x = upsample_block (
        x ,
        128 ,
        layers.ELU ( 0.2 ) ,
        strides=(1 , 1) ,
        use_bias=False ,
        use_bn=False ,
        padding="same" ,
        use_dropout=False ,
    )
    x = upsample_block (
        x ,
        128 ,
        layers.ELU ( 0.2 ) ,
        strides=(1 , 1) ,
        use_bias=False ,
        use_bn=False ,
        padding="same" ,
        use_dropout=False ,
    )
    x = upsample_block (
        x ,
        128 ,
        layers.ELU ( 0.2 ) ,
        strides=(1 , 1) ,
        use_bias=False ,
        use_bn=False ,
        padding="same" ,
        use_dropout=False ,
    )
    x = upsample_block (
        x , 1 , layers.Activation ( "tanh" ) , strides=(1 , 1) , use_bias=False , use_bn=True
    )
    x = layers.Cropping2D ( (7 , 7) ) ( x )

    g_model = keras.models.Model ( noise , x , name="generator" )
    return g_model


g_model = get_generator_model ( )


class WGAN ( keras.Model ):
    def __init__ (
            self ,
            discriminator ,
            generator ,
            latent_dim ,
            discriminator_extra_steps=3 ,
            gp_weight=10.0 ,
    ):
        super ( WGAN , self ).__init__ ( )
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile ( self , d_optimizer , g_optimizer , d_loss_fn , g_loss_fn ):
        super ( WGAN , self ).compile ( )
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty ( self , batch_size , real_images , fake_images ):
        alpha = tf.random.normal ( [ batch_size , 1 , 1 , 1 ] , 0.0 , 1.0 )
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape ( ) as gp_tape:
            gp_tape.watch ( interpolated )
            pred = self.discriminator ( interpolated , training=True )

        grads = gp_tape.gradient ( pred , [ interpolated ] ) [ 0 ]
        norm = tf.sqrt ( tf.reduce_sum ( tf.square ( grads ) , axis=[ 1 , 2 , 3 ] ) )
        gp = tf.reduce_mean ( (norm - 1.0) ** 2 )
        return gp

    def train_step ( self , real_images ):
        if isinstance ( real_images , (tuple , list) ):
            real_images = real_images [ 0 ]

        batch_size = tf.shape ( real_images ) [ 0 ]

        for i in range ( self.d_steps ):
            random_latent_vectors = tf.random.normal (
                shape=(batch_size , self.latent_dim) )

            with tf.GradientTape ( ) as tape:
                fake_images = self.generator ( random_latent_vectors , training=True )
                fake_logits = self.discriminator ( fake_images , training=True )
                real_logits = self.discriminator ( real_images , training=True )
                d_cost = self.d_loss_fn ( real_img=real_logits , fake_img=fake_logits )
                gp = self.gradient_penalty ( batch_size , real_images , fake_images )
                d_loss = d_cost + gp * self.gp_weight
            d_gradient = tape.gradient ( d_loss , self.discriminator.trainable_variables )
            self.d_optimizer.apply_gradients (
                zip ( d_gradient , self.discriminator.trainable_variables )
            )

        random_latent_vectors = tf.random.normal ( shape=(batch_size , self.latent_dim) )
        with tf.GradientTape ( ) as tape:
            generated_images = self.generator ( random_latent_vectors , training=True )
            gen_img_logits = self.discriminator ( generated_images , training=True )
            g_loss = self.g_loss_fn ( gen_img_logits )

        gen_gradient = tape.gradient ( g_loss , self.generator.trainable_variables )
        self.g_optimizer.apply_gradients (
            zip ( gen_gradient , self.generator.trainable_variables )
        )
        return {"d_loss": d_loss , "g_loss": g_loss}


class GANMonitor ( keras.callbacks.Callback ):
    def __init__ ( self , num_img=25 , latent_dim=320 ):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end ( self , epoch , logs=None ):
        random_latent_vectors = tf.random.normal ( shape=(self.num_img , self.latent_dim) )
        generated_images = self.model.generator ( random_latent_vectors )
        generated_images = (generated_images * 127.5) + 127.5

        for i in range ( self.num_img ):
            img = generated_images [ i ].numpy ( )
            img = keras.preprocessing.image.array_to_img ( img )
            img.save ( path + '{epoch}_{i}.png'.format ( i=i + 1 , epoch=epoch + 1 ) )


generator_optimizer = keras.optimizers.Adam ( learning_rate=0.0002 , beta_1=0.5 , beta_2=0.9 , decay=1e-4 )
discriminator_optimizer = keras.optimizers.Adam ( learning_rate=0.0002 , beta_1=0.5 , beta_2=0.9 , decay=1e-4 )


def discriminator_loss ( real_img , fake_img ):
    real_loss = tf.reduce_mean ( real_img )
    fake_loss = tf.reduce_mean ( fake_img )
    return fake_loss - real_loss


def generator_loss ( fake_img ):
    return -tf.reduce_mean ( fake_img )


epochs = 1000
gp_weight = 10
discriminator_extra_steps = 3
path = 'somewhere/somewhere'
checkpoint_filepath = os.path.join ( path , 'weights_{epoch}.hdf5' )

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint (
    filepath=checkpoint_filepath ,
    save_weights_only=True ,
    save_best_only=False , verbose=1 , save_freq='epoch' )

cbk = GANMonitor ( num_img=15 , latent_dim=noise_dim )

wgan = WGAN ( discriminator=d_model , generator=g_model , latent_dim=noise_dim ,
              discriminator_extra_steps=discriminator_extra_steps , gp_weight=gp_weight )

wgan.compile ( d_optimizer=discriminator_optimizer ,
               g_optimizer=generator_optimizer ,
               g_loss_fn=generator_loss ,
               d_loss_fn=discriminator_loss )

wgan.built = True
wgan.load_weights ( r'W:\Deepkoagans/weights_647.hdf5' )
