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
import keras.backend as K
import matplotlib.pyplot as plt
from scipy.stats import itemfreq

import sys
from tqdm import tqdm
import kdd as data

import numpy as np

from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, auc
from sklearn.metrics import precision_recall_fscore_support

class GAN():
    def __init__(self):

        self.learning_rate =  0.00001
        self.batch_size = 50
        self.latent_dim = 32
        self.dis_inter_layer_dim = 128
        #init_kernel = initializers.random_normal(mean=0.0, stddev=0.02)
        self.RANDOM_SEED = 146
        #FREQ_PRINT = 20 # print frequency image tensorboard [20]
        self.init_kernel = initializers.glorot_uniform()
        self.inital_center = 0
        #Loading the data
        self.X_train, self.y_train = data.get_train()
        #trainx_copy = trainx.copy()
        self.X_test, self.y_test = data.get_test()
        self.data_shape = data.get_shape_input()
    
    
    def adam_optimizer(self):
        return keras.optimizers.Adam(lr=self.learning_rate , beta_1=0.5)
    
        
    def build_generator(self):
        
        generator = Sequential()
        generator.add(Dense(64, kernel_initializer=self.init_kernel,input_dim=self.latent_dim))
        generator.add(LeakyReLU())
        generator.add(Dense(128, kernel_initializer=self.init_kernel))
        generator.add(LeakyReLU())
        generator.add(Dense(121))
        #generator.compile(loss='binary_crossentropy', optimizer=self.adam_optimizer())
        generator.summary()
        noise = Input(shape=(self.latent_dim,))
        img = generator(noise)
        return Model (noise, img)
    
    
    def build_discriminator(self):
        
        discriminator = Sequential()
        discriminator.add(Dense(256, input_dim=121, kernel_initializer=self.init_kernel))
        discriminator.add(LeakyReLU(0.1))
        discriminator.add(Dropout(0.2))
        
        discriminator.add(Dense(128))
        discriminator.add(LeakyReLU(0.1))
        discriminator.add(Dropout(0.2))
        
        discriminator.add(Dense(self.dis_inter_layer_dim))
        discriminator.add(LeakyReLU(0.1))
        discriminator.add(Dropout(0.2))
        
        #intermediate_layer = discriminator
        
        discriminator.add(Dense(1, activation='sigmoid'))
        #discriminator.compile(loss='binary_crossentropy', optimizer=self.adam_optimizer())
        
        discriminator.summary()

        
        return discriminator


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

        #valid = np.ones((self.batch_size, 1))
        #fake = np.zeros((self.batch_size, 1))
       
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
                #y_dis = np.concatenate([valid, fake])
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
        
        

    
    def test_auc(self, discriminator, x_test,y_test):
        score = discriminator.predict(x_test)
        
        per = np.percentile(score, 80)
        print(per)
        
         
        y_pred = score .copy()
        y_pred = np.array(y_pred)
        #y_pred = np.where(y_pred >= per, 0, 1)
        
        inds = (y_pred <= per)
        inds_comp = (y_pred > per)
        
        y_pred[inds] = 0
        y_pred[inds_comp] = 1
        
        print(y_pred[:50])
        print(y_test[:50])
        print(score[:50])
        
        print(itemfreq(y_test))
        print(itemfreq(y_pred))
        
        precision, recall, f1,_ = precision_recall_fscore_support(y_test,y_pred, average='binary')
        print(
            "Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f "
            % (precision, recall, f1))

if __name__ == '__main__':
    gan = GAN()
    gan.train_test(epochs=5)

#note
#applied statified in the spliting process. 

#GAN :  Prec: 0.9184  Rec: 0.9325   F1: 0.9254
#gan Fm : Prec: 0.9355  Rec: 0.9504   F1: 0.9429
#Lessin 0.9846869199683281

