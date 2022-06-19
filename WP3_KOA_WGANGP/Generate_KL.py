#%% Generate Images
batch_gen = 32
dirname='Fakes'
# os.mkdir(r'W:\Deepkoagans\Experiment_RETRAIN23KL02_withdropout23/'+str(dirname))
for b in range ( 0 , batch_gen ):
    img_n = 100
    noise = noise_dim
    random_latent_vectors = tf.random.normal ( shape=(img_n , noise) )
    generated_images = wgan.generator ( random_latent_vectors )
    generated_images = (generated_images * 127.5) + 127.5

    for i in range ( img_n ):
        img = generated_images [ i ].numpy ( )
        img = keras.preprocessing.image.array_to_img ( img )
        img.save ( (r'W:/'+str(dirname)+'\gen_img_kl023_{b}_{i}.png').format ( i=i , b=b ) )
    print ( 'batch_ready:' , b )

