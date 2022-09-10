# -*- coding: utf-8 -*-
"""
Created on Thu May 12 22:20:20 2022

@author: Orlando
"""
import numpy as np
import pandas as pd

from keras.datasets.mnist import load_data
(x_train, y_train), (x_test, y_test) = load_data()

print("train", x_train.shape, y_train.shape)
print("test", x_test.shape, y_test.shape)

def load_012():
  (x_train,y_train), (_, _) = load_data()
  zz=np.zeros(len(y_train))
  idxz = np.where(y_train==zz)
  oo=np.ones(len(y_train))
  idxo = np.where(y_train==oo)
  dd=2*np.ones(len(y_train))
  idxd = np.where(y_train==dd)

  index = np.hstack([idxz[0],idxo[0],idxd[0]])
  sub_train = x_train[index]
  return sub_train,y_train[index]

from matplotlib import pyplot
x_train,y_train = load_012()
print(len(x_train))
i=0
x=np.random.choice(range(len(x_train)), 36)
for index in x:
  pyplot.subplot(6,6,1+i)
  i+=1
  pyplot.axis('off')
  pyplot.imshow(x_train[index], cmap='gray_r')
  
  
#discriminator :
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU

def define_discriminator(in_shape=(28,28,1)):
  model = Sequential()
  model.add(Conv2D(16, (7,7), strides=(1,1), padding='same', input_shape=in_shape))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Conv2D(10, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

  return model

model = define_discriminator()

#defining data loading functions : 
from numpy.random.mtrand import randint, rand
from numpy import expand_dims
def load_dataset():
  x_train,_ = load_012()
  X = expand_dims(x_train, axis=-1)
  X = X.astype('float32')
  X = X/255.0
  return X

def generate_real_samples(dataset, n_samples):
  #get random indices
  idx = randint(0, dataset.shape[0], n_samples)
  #sample according to indices
  X = dataset[idx]
  #create labels
  y = np.ones((n_samples, 1))
  return X,y

def generate_fake_samples(n_samples):
  #rand is already between 0 and 1
  X = rand(28*28*n_samples)
  X = X.reshape((n_samples, 28, 28,1))
  y = np.zeros((n_samples, 1))
  return X,y
#j'ai essayé d'entrainer le disc, ça marche.

#générateur : 
from keras.layers import Reshape
from keras.layers import Conv2DTranspose

def define_generator(latent_dim):
  model = Sequential()
  #transform the latent space into 10 7x7 low-res images
  model.add(Dense(10*7*7, input_dim=100))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Reshape((7,7,10)))
  #conv2DTranspose is the same as upsampling followed by convolution
  model.add(Conv2DTranspose(10, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2DTranspose(10, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  #make the final image
  model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
  return model

#generator utils functions
import numpy.matlib

def make_vectors(latent_dim, n_samples):
  samples = np.matlib.randn(n_samples * latent_dim)
  return samples.reshape(n_samples, latent_dim)

def generate_generator_images(g_model,n_samples, latent_dim):
  vectors = make_vectors(latent_dim, n_samples)
  X = g_model.predict(vectors)
  #sans ce reshape on a du 28,28,1 pour les images
  X = X.reshape((n_samples,28,28))
  y=np.zeros((n_samples, 1))
  return X,y

def generate_generator_samples(g_model,n_samples, latent_dim):
  vectors = make_vectors(latent_dim, n_samples)
  X = g_model.predict(vectors)
  y=np.zeros((n_samples, 1))
  print("shape of X : ",X.shape)
  return X,y


#GAN
def define_gan(g_model, d_model):
  d_model.trainable=False
  model = Sequential()
  model.add(g_model)
  model.add(d_model)
  #Adam beta_1 : initial decay rate (so how the lr changes ?)
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)
  return model

#entrainement :
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
  bat_per_epo = int(int(dataset.shape[0])/n_batch)
  half_batch = int(n_batch/2)
  for i in range(n_epochs):
    for j in range(bat_per_epo):
      X_real, y_real = generate_real_samples(dataset, half_batch)
      X_fake, y_fake = generate_generator_samples(g_model, half_batch, latent_dim)
      X= np.vstack((X_real, X_fake))
      y=np.vstack((y_real, y_fake))
      d_loss,_ = d_model.train_on_batch(X,y)

      X_gan = make_vectors(latent_dim, n_batch)
      y_gan = np.ones((n_batch,1))

      g_loss = gan_model.train_on_batch(X_gan, y_gan)
      print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
    X,y = generate_generator_images(g_model, 20, 100)
    if i%10==0:
        for k in range(16):
          pyplot.subplot(4,4,1+k)
          pyplot.axis('off')
          pyplot.imshow(X[k], cmap='gray_r')
          pyplot.show()
         
#lancer l'expérimentation
d_model = define_discriminator()
g_model = define_generator(100)
gan_model = define_gan(g_model, d_model)

dataset = load_dataset()
train(g_model,d_model, gan_model, dataset, 100, n_epochs=100)
