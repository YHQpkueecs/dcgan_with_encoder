# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:44:08 2019

@author: Yhq
"""

import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from model import get_generator, get_discriminator, get_encoder
from data import flags, get_celebA

n_epoch = 20
batch_size = 64
learning_rate = 0.0004
save_every_epoch = 4
print_every_step = 20
G_weights = flags.checkpoint_dir + '/G_20.h5'


def train():
    images, images_path = get_celebA(flags.output_size, flags.n_epoch, flags.batch_size)
    G = get_generator()
    E = get_encoder()
    G.load_weights(G_weights)
    G.train()
    E.train()
    optimizer = tf.optimizers.Adam(learning_rate, beta_1=flags.beta1)
    
    n_step_epoch = int(len(images_path) // flags.batch_size)
    
    for epoch in range(n_epoch):
        for step, batch_images in enumerate(images):
            if batch_images.shape[0] != flags.batch_size:
                break
            
            step_time = time.time()
            with tf.GradientTape() as tape:
                z = np.random.normal(loc=0.0, scale=1.0, size=[batch_size, flags.z_dim]).astype(np.float32)
                gen = G(z)
                z_encode = E(gen)
                
                x_encode = E(batch_images)
                x_decode = G(x_encode)
                
                z_recon_loss = tl.cost.absolute_difference_error(z_encode, z, is_mean=True)
                x_recon_loss = 5. * tl.cost.absolute_difference_error(x_decode, batch_images, is_mean=True)
                loss = z_recon_loss + x_recon_loss
                
            grad = tape.gradient(loss, E.trainable_weights)
            optimizer.apply_gradients(zip(grad, E.trainable_weights))
            
            if step % print_every_step == 0:
                print('Epoch: [{}/{}] step: [{}/{}] took: {:3f}, z_recon_loss: {:5f}, x_recon_loss: {:5f}'.format(epoch, n_epoch,\
                      step, n_step_epoch, time.time()-step_time, z_recon_loss, x_recon_loss))
        
        if epoch % save_every_epoch == 0:
            E.save_weights('{}/E_{}.h5'.format(flags.checkpoint_dir, epoch))
        

if __name__=='__main__':
    train()