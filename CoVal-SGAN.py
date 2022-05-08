# -*- coding: utf-8 -*-
"""
This script implements the CoVal-SGAN proposed in:

@InProceedings{Scarpiniti2022,
  author    = {Michele Scarpiniti, Cristiano Mauri, Danilo Comminiello, Aurelio Uncini},
  booktitle = {2022 International Joint Conference on Neural Networks (IJCNN 2022)},
  title     = {CoVal-SGAN: A Complex-Valued Spectral GAN architecture for the effective audio data augmentation in construction sites},
  year      = {2022},
  address   = {Padova, Italy},
  pages     = {1--8},
  doi       = {},
}

Specifically, the proposed CoVal-SGAN is used to generate new synthetic spectrogram to augment
audio data of the more problematic among the considered class. The augemnted dataset is used
for the classification of different equipments on construction sites.

The complex-valued implementation of the proposed GAN exploits the following software package,
which should be installed:

@software{j_agustin_barrachina_2021_4452131,
  author       = {J Agustin Barrachina},
  title        = {Complex-Valued Neural Networks (CVNN)},
  month        = jan,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.6},
  doi          = {10.5281/zenodo.4452131},
  url          = {https://doi.org/10.5281/zenodo.4452131}
}

See also: https://github.com/NEGU93/cvnn


The implementation of complex batch normmalization is that proposed in:
    
    @ARTICLE {,
    author  = "Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal",
    title   = "Deep Complex Networks",
    journal = "arXiv preprint arXiv:1705.09792",
    year    = "2017"
}
    
    This implementation is available here:
    https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py


Created on Wed Apr 14 15:39:21 2021

@author: Cristiano Mauri and Michele Scarpiniti

"""

import numpy as np
import cvnn.layers as complex_layers
import tensorflow as tf
from tensorflow.keras import layers
import librosa as lb
import librosa.display
import time
import matplotlib.pyplot as plt
import skimage.io


# %% Creating the dataset

sample_path = "./DATA"

frame_size = 30  # Frame size in ms 
n_fft = 1024     # Number of FFT bins
hop_length = 512 # Number of samples per time-step in spectrogram
n_mels = 128     # Number of bins in spectrogram. Height of image

print('Loading wav file...', end='')

# Loading audio data related to the JD50D class
y, samplerate = lb.load(sample_path+"/JD50D_A_onsite.wav")
frame_length  = int(samplerate*frame_size/1000+1)

mel = lb.filters.mel(sr=samplerate, n_fft=n_fft, n_mels=n_mels) #scala di Mel per i parametri scelti

print('...Done!')

step = int((frame_length)/2) + 1  # 50% overlap
stop = y.size - frame_length

n = 0
dataset_tensor = []

print('Creating dataset...')

for i in range(0, stop, step):
    fft_window = lb.stft(y[i:i+frame_length], n_fft=n_fft, hop_length=hop_length)
    dataset_tensor.append(fft_window)
    if n % 1000 == 0:
        print('.', end='')
    n+=1

# Creating TF dataset
dataset_cast = tf.cast(dataset_tensor, tf.complex64)
dataset_fin  = tf.data.Dataset.from_tensor_slices(tf.expand_dims(dataset_cast, axis=-1))

print('Done!')

num_elements = tf.data.experimental.cardinality(dataset_fin).numpy()

print(f'{num_elements} elements in the dataset.')
print('')



# %% Creating the CoVal-SGAN

import cvnn
from bn import ComplexBatchNormalization


complex_leaky_relu = layers.Lambda(lambda z: tf.cast(tf.complex(tf.nn.leaky_relu(tf.math.real(z), 0.3, None),
                              tf.nn.leaky_relu(tf.math.imag(z), 0.3, None)), dtype=z.dtype))

complex_relu = layers.Lambda(lambda z: tf.cast(tf.complex(tf.keras.activations.relu(tf.math.real(z), 0.0, None, 0),
                              tf.keras.activations.relu(tf.math.imag(z), 0.0, None, 0)), dtype=z.dtype))

softmax_real_with_abs = layers.Lambda(lambda z: tf.keras.activations.softmax(tf.math.abs(z), axis=-1))

initializer = cvnn.initializers.ComplexGlorotUniform()


# Define the generator
def Generator(dim=64):
    # Create your model
    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=(100,)))
 
    model.add(complex_layers.ComplexDense(19*1*dim*4, use_bias=False))
    model.add(ComplexBatchNormalization()) 
    model.add(complex_leaky_relu)
    model.add(layers.Reshape((19, 1, dim*4)))   
    
    model.add(complex_layers.ComplexConv2DTranspose(dim*2, (6, 2), strides=(3, 1), padding='same', use_bias=False))
    model.add(ComplexBatchNormalization())
    model.add(complex_leaky_relu)
    
    model.add(complex_layers.ComplexConv2DTranspose(dim, (6, 2), strides=(3, 2), padding='same', use_bias=False))
    model.add(ComplexBatchNormalization())
    model.add(complex_leaky_relu)
    
    model.add(complex_layers.ComplexConv2DTranspose(1, (6, 2), strides=(3, 1), padding='same', use_bias=False))   
    # No tanh activation if input is not normalized
    
    return model

    
# Define the discriminator
def Discriminator(dim=64):
    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=(513, 2, 1)))
    
    model.add(complex_layers.ComplexConv2D(dim, (5, 5), strides=(3, 2), padding='same'))
    model.add(complex_leaky_relu)
    model.add(complex_layers.ComplexDropout(0.3))
    
    model.add(complex_layers.ComplexConv2D(dim*2, (5, 5), strides=(3, 1), padding='same'))
    model.add(complex_leaky_relu)
    model.add(complex_layers.ComplexDropout(0.3))

    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(1, activation='convert_to_real_with_abs'))
    return model    
# An activation that casts to real must be used at the last layer. 
# The loss function cannot minimize a complex number


# Create the generator and discriminator of the CoVal-SGAN
gen  = Generator()
disc = Discriminator()

gen.summary()
disc.summary()


#======================================================================================
#                        Losses
#======================================================================================

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()


# Define the discriminator loss
def discriminator_loss(real_output, fake_output, loss):
    
    if loss == 'dcgan':
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        
    elif loss == 'lsgan':
        real_loss = mse(tf.ones_like(real_output), real_output)
        fake_loss = mse(tf.zeros_like(fake_output), fake_output)
    
    elif loss == 'ggan': # Original Goodfellow et al. GAN
        real_loss = -tf.math.log(real_output)*2
        fake_loss = -tf.math.log(1-fake_output)*2
    
    total_loss = real_loss + fake_loss
        
    return total_loss*0.5


# Define the discriminator loss
def generator_loss(fake_output, loss):
    
    if loss == 'dcgan':
        g_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    elif loss == 'lsgan':
        g_loss = mse(tf.ones_like(fake_output), fake_output)
        
    elif loss == 'ggan':
        g_loss = -tf.math.log(fake_output)
        
    return g_loss



# %% Training the CoVal-SGAN

EPOCHS = 200

learning_rate = 0.0001

noise_dim = 100
batch_size = 128

data_perc = 100 # Percentage of the dataset to be used

dataset = dataset_fin.cache().shuffle(20000).take(data_perc*num_elements/100).batch(batch_size, drop_remainder=True)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) #beta1 = 0.5 for dcgan
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


#Create test complex noise vectors
noise_real = tf.cast(tf.random.normal([batch_size, noise_dim], 0.0, 5.0), tf.complex64)
noise_imag = tf.cast(tf.random.normal([batch_size, noise_dim], 0.0, 5.0), tf.complex64)
unit_imag  = tf.cast(tf.constant(1j), tf.complex64)
noise_in   = tf.add(noise_real, tf.multiply(noise_imag, unit_imag))
sample_spec = next(iter(dataset)) # Test sample for comparison


#==================================================================================================================
#                                       Training
#==================================================================================================================

@tf.function
def train_step(images):
    #Create complex noise vector
    noise_real = tf.cast(tf.random.normal([batch_size, noise_dim], 0.0, 5.0), tf.complex64)
    noise_imag = tf.cast(tf.random.normal([batch_size, noise_dim], 0.0, 5.0), tf.complex64)
    unit_imag  = tf.cast(tf.constant(1j), tf.complex64)
    
    noise = tf.add(noise_real, tf.multiply(noise_imag, unit_imag))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise, training=True)
    
        real_output = disc(images, training=True)
        fake_output = disc(generated_images, training=True)
    
        gen_loss  = generator_loss(fake_output, loss='lsgan')
        disc_loss = discriminator_loss(real_output, fake_output, loss='lsgan')

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
    
    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, test_input, sample_spec):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    predictions_array = np.array(predictions[0, :, :, 0])
    predictions_abs = np.abs(predictions_array) ** 2
    predictions_mel = mel.dot(predictions_abs)
    predictions_logmel = np.log(predictions_mel+1e-9)
    # real_pred = np.real(predictions_array)
    # imag_pred = np.imag(predictions_array)    
    
    sample_array = np.array(sample_spec[0, :, :, 0])
    sample_abs = np.abs(sample_array) ** 2
    sample_mel = mel.dot(sample_abs)
    sample_logmel = np.log(sample_mel+1e-9)
    # real_sample = np.real(sample_array)
    # imag_sample = np.imag(sample_array)    
  
    plt.figure(figsize=(32,16))
    plt.subplot(5, 5, 1)
    lb.display.specshow(sample_logmel, sr=samplerate, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title("Original Sample")
    plt.colorbar(format='%+2.0f dB')
    plt.subplot(5, 5, 2)
    lb.display.specshow(predictions_logmel, sr=samplerate, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title("Prediction")
    plt.colorbar(format='%+2.0f dB')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    
    
    # If uncommented, real and imaginary parts of the generated samples are plotted
    # import librosa.display as lbdp
    
    # plt.figure()
    # lb.display.specshow(real_pred, sr=samplerate, hop_length=hop_length, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title("Real Spectrogram "+str(epoch))
    
    # plt.figure()
    # lb.display.specshow(imag_pred, sr=samplerate, hop_length=hop_length, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title("Imag Spectrogram "+str(epoch))
  
    plt.show()


def train(dataset, epochs):
    for epoch in range(epochs):
          
        n=0
        start = time.time()
      
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            
            if n % 1 == 0:
                print('.', end='')
            n+=1
      
        # Produce images as we go
        if (epoch+1) % 5 == 0:
            generate_and_save_images(gen, epoch+1, test_input=noise_in, sample_spec=sample_spec) # Usa "sample_spec" per UNET
            gen.save_weights('./checkpoints/genEpoch'+str(epoch+1)+'.h5')
      
        # Uncomment if loss == 'ggan', to avoid multiple visualization of the loss for each batch
        # gen_loss_mean = tf.reduce_sum(gen_loss) * (1. / batch_size)
        # disc_loss_mean = tf.reduce_sum(disc_loss) * (1. / batch_size)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start), end='')
        print (f'[G Loss: {gen_loss}] ', end='')
        print (f'[D Loss: {disc_loss}] ')
    
    
# Train the model
train(dataset, EPOCHS)

    
# %% Generate and plot a new sample

# Define a function for normalization in [-1, +1]
def scale_minmax(X, min=-1.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


predictions = gen(noise_in, training=False)
predictions_array = np.array(predictions[0, :, :, 0])
predictions_abs = np.abs(predictions_array) ** 2
predictions_mel = mel.dot(predictions_abs)
predictions_logmel = np.log(predictions_mel+1e-9)
predictions_scaled = scale_minmax(predictions_logmel, min=-12.5, max=2.2)

plt.figure()
lb.display.specshow(predictions_scaled, sr=samplerate, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Scaled Spectrogram "+str(EPOCHS))

plt.show()


# %% Generate new synthetic samples

NUM_GEN_SAMPLES = 4000

data_dir = './Generated_Samples/JD50D'

print("Generation of "+ str(NUM_GEN_SAMPLES) +" samples: ")

gen.load_weights('./checkpoints/genEpoch'+str(EPOCHS)+'.h5')

for i in range(NUM_GEN_SAMPLES):
    noise_real = tf.cast(tf.random.normal([batch_size, noise_dim], 0.0, 5.0), tf.complex64)
    noise_imag = tf.cast(tf.random.normal([batch_size, noise_dim], 0.0, 5.0), tf.complex64)
    unit_imag  = tf.cast(tf.constant(1j), tf.complex64)
    
    seed = tf.add(noise_real, tf.multiply(noise_imag, unit_imag))
    predictions = gen(seed, training=False)
    predictions_array = np.array(predictions[0, :, :, 0])
    predictions_abs = np.abs(predictions_array) ** 2
    predictions_mel = mel.dot(predictions_abs)
    predictions_logmel = np.log(predictions_mel+1e-9)
    predictions_scaled = scale_minmax(predictions_logmel, min=0, max=255)
    img = np.array(predictions_scaled)
    img_flip = np.flip(img, axis=0)
    I = img_flip.astype(np.uint8)
    skimage.io.imsave(data_dir + "\output"+ str(i) +"_generator_JD50D.png", I)
    if (i+1) % (NUM_GEN_SAMPLES/100) == 0:
        print ('.', end='')
    
print("\nDone!")
