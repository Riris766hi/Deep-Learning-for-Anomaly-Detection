from __future__ import print_function, division

import keras
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import initializers

import matplotlib.pyplot as plt

import sys
from tqdm import tqdm


import numpy as np
import mnist as data
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, auc
from sklearn.metrics import precision_recall_fscore_support

class GAN():
    def __init__(self):

        self.learning_rate = 0.00001
        self.batch_size = 100
        self.latent_dim = 100
        self.dis_inter_layer_dim = 1024
        #init_kernel = initializers.random_normal(mean=0.0, stddev=0.02)
        self.RANDOM_SEED = 146
        #FREQ_PRINT = 20 # print frequency image tensorboard [20]
        self.init_kernel = initializers.RandomNormal(stddev=0.02)
        self.inital_center = 0
        #Loading the data
        self.X_train, self.y_train = data.get_train(0, True)
        #trainx_copy = trainx.copy()
        self.X_test, self.y_test = data.get_test(0, True)



    def adam_optimizer(self):
        return keras.optimizers.Adam(lr=self.learning_rate , beta_1=0.5)


    def build_generator(self):
            
        generator=Sequential()
        generator.add(Dense(1024, kernel_initializer=self.init_kernel,input_dim=self.latent_dim,
                            activation='relu'))
        generator.add(BatchNormalization())
        generator.add(Dense(7*7*128, kernel_initializer=self.init_kernel,
                            activation='relu'))
        generator.add(BatchNormalization())
        generator.add(Reshape((7, 7, 128)))
        
        generator.add(Conv2DTranspose(64, (4, 4),strides=(2, 2),  padding='same',
                            kernel_initializer=self.init_kernel,activation='relu'))
        generator.add(BatchNormalization())
        generator.add(Conv2DTranspose(1,(4, 4),strides=(2, 2),  padding='same',
                            kernel_initializer=self.init_kernel ,activation='tanh'))
        
        generator.summary()
        noise = Input(shape=(self.latent_dim,))
        img = generator(noise)
        return Model (noise, img)
        

    def build_discriminator(self):
        
        input_shape = (28, 28, 1)
        
        discriminator=Sequential()
        discriminator.add(Conv2D(64, 4, padding='same', strides=2, input_shape=input_shape,
                                 kernel_initializer=self.init_kernel))
        discriminator.add(LeakyReLU(0.1))
        discriminator.add(Conv2D(64, 4, padding='same', strides=2,
                                 kernel_initializer=self.init_kernel))
        discriminator.add(LeakyReLU(0.1))
        #discriminator.add(Reshape((7, 7, 64)))
        discriminator.add(Flatten())
        discriminator.add(Dense(1024, kernel_initializer=self.init_kernel))
        discriminator.add(LeakyReLU(0.1))
        discriminator.add(Dense(1,kernel_initializer=self.init_kernel, activation='sigmoid'))
        
        discriminator.summary()
        img = Input(shape=input_shape)
        validity = discriminator(img)

        return Model(img, validity)
        

    
    def get_gan_network(self, discriminator, generator):
        # We initially set trainable to False since we only want to train either the
        # generator or discriminator at a time
        discriminator.trainable = False
        # gan input (noise) will be 100-dimensional vectors
        gan_input = Input(shape=(self.latent_dim,))
        # the output of the generator (an image)
        x = generator(gan_input)
        # get the output of the discriminator (probability if the image is real or not)
        gan_output = discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=self.adam_optimizer())
        gan.summary()
        return gan


        
    def train_test(self, epochs):
        
        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy',
                                   optimizer=self.adam_optimizer(),
                                   metrics=['accuracy'])
                                   
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.adam_optimizer(),
                                   metrics=['accuracy'])
            
        
        # Split the training data into batches of size 128
        batch_count = self.X_train.shape[0] / self.batch_size
        
        anomaly_label = np.ones((self.batch_size, 1))
        normal_label = np.zeros((self.batch_size, 1))

        # Build our GAN netowrk
        
        # Build our GAN netowrk
        gan = self.get_gan_network(self.discriminator, self.generator)

        for e in range(1, epochs+1):
        #print("Epoch %d" %e)
            for _ in tqdm(range(int(batch_count))):
                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[self.batch_size, self.latent_dim])
                image_batch = self.X_train[np.random.randint(0, self.X_train.shape[0], size=self.batch_size)]

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images


                # Generate fake date
                generated_images = self.generator.predict(noise)
                
                #self.discriminator.train_on_batch(image_batch, normal_label)
                #self.discriminator.train_on_batch(generated_images, anomaly_label)

                X = np.concatenate([image_batch, generated_images])
                y_dis=np.zeros(2*self.batch_size)
                y_dis[:self.batch_size]=0.9

                # Labels for generated and real data
                #y_dis =  np.array([0] * self.batch_size + [1] * self.batch_size)
                # One-sided label smoothing
                #y_dis[:self.batch_size] = 0.9
                # Labels for generated and real data
                #y_dis=  np.array([0] * self.batch_size + [1] * self.batch_size)
                #y_dis = np.concatenate([normal_label, anomaly_label])
                #y_dis[:self.batch_size]=0.9
                self.discriminator.trainable = True
                self.discriminator.train_on_batch(X,y_dis)
                # Train generator
                noise = np.random.normal(0, 1, size=[self.batch_size, self.latent_dim])
                y_gen = np.ones(self.batch_size)
                self.discriminator.trainable = False
                gan.train_on_batch(noise,y_gen )
                

                                                   # Plot the progress
                #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (e, d_loss[0], 100*d_loss[1], g_loss))
        print("Epoch %d" %e)
        self.test_auc(self.discriminator,self.X_test,self.y_test)
        print(self.y_test[:50])
        

        
        
    def test_auc(self, discriminator ,x_data,x_label):
        scorer = discriminator.predict(x_data)
        print(scorer[:50])
        precision, recall, thresholds = precision_recall_curve(x_label, scorer)
        prc_auc = auc(recall, precision)
        print("AUPRC: ", prc_auc)


if __name__ == '__main__':
    gan = GAN()
    gan.train_test(epochs=2)


    #AUPRC:  0.6277425691472485
    #fm :  0.5662051311010794


