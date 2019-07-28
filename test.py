# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:40:58 2019

@author: Yhq
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import get_generator, get_encoder
from data import get_celebA
from tqdm import trange

tl.logging.set_verbosity(tl.logging.FATAL)

class FLAGS(object):
    def __init__(self):
        self.z_dim = 100 
        self.output_size = 64 
        self.c_dim = 3 
        self.checkpoint_dir = "checkpoint"
        
flags = FLAGS() 
g_weights = 'checkpoint/G_20.h5'
e_weights = 'checkpoint/E_8.h5'

# load models
G = get_generator()
G.load_weights(g_weights)
G.eval()
E = get_encoder()
E.load_weights(e_weights)
E.eval()

# randomly generate faces
def gen(n=16):
    z = np.random.normal(loc=0.0, scale=1.0, size=[n, flags.z_dim]).astype(np.float32)
    gen = G(z)
    gen = np.array((gen + 1.) * 127.5, dtype=np.uint8)
    for i in range(n):
        plt.imshow(gen[i])
        plt.show()


def vec_interpo(v1, v2, a):
    return a * v2 + (1. - a) * v1

# interpolate between two random faces
def interpolate_rand(n=16, drc = 'interp'):
    z1 = np.random.normal(loc=0.0, scale=1.0, size=[1, flags.z_dim]).astype(np.float32)
    z2 = np.random.normal(loc=0.0, scale=1.0, size=[1, flags.z_dim]).astype(np.float32)
    
    ims = []
    for i, a in zip([j for j in range(n)], np.linspace(0., 1., n)):
        z = vec_interpo(z1, z2, a)
        gen = G(z)
        ims.append(np.array(gen, dtype=np.float32))
        gen = np.array((gen + 1.) * 127.5, dtype=np.uint8).reshape((flags.output_size, flags.output_size, flags.c_dim))
        #plt.imshow(gen)
        #plt.show()
        Image.fromarray(gen).save('{}/{}.jpg'.format(drc, i))
    ims = np.asarray(ims).reshape((n,64,64,3))
    tl.visualize.save_images(ims, [1, n], '{}/vis.png'.format(drc))
 
 
# reconstruct image
def recon():
    ds, _ = get_celebA(flags.output_size, 1, 1)
    for im in ds:
        if input() == 'q':
            break
        #print(im)
        rec = G(E(im))
        plt.imshow(np.array((im + 1.) * 127.5, dtype=np.uint8)[0])
        plt.show()
        plt.imshow(np.array((rec + 1.) * 127.5, dtype=np.uint8)[0])
        plt.show()

# interpolate between two selected faces
def interpolate(file1, file2, n=16, drc='interp_select'):
    im1, im2 = Image.open(file1).crop((25, 45, 153, 173)).resize((64, 64)),\
                    Image.open(file2).crop((25, 45, 153, 173)).resize((64, 64))
    #plt.imshow(im)
    #plt.show() 
    im1, im2 = np.array(im1, dtype=np.float32).reshape((1,64,64,3)), \
                    np.array(im2, dtype=np.float32).reshape((1,64,64,3))
    im1, im2 = im1 / 127.5 - 1, im2 / 127.5 - 1
    
    z1 = np.array(E(im1), dtype=np.float32)
    z2 = np.array(E(im2), dtype=np.float32)
    
    '''
    rec = G(E(im2))
    plt.imshow(np.array((im2 + 1.) * 127.5, dtype=np.uint8)[0])
    plt.show()
    plt.imshow(np.array((rec + 1.) * 127.5, dtype=np.uint8)[0])
    plt.show()
    '''
    ims = []
    ims.append(im1)
    for i, a in zip([j for j in range(n)], np.linspace(0., 1., n)):
        z = vec_interpo(z1, z2, a)
        gen = G(z)
        ims.append(np.array(gen, dtype=np.float32))
        gen = np.array((gen + 1.) * 127.5, dtype=np.uint8).reshape((flags.output_size, flags.output_size, flags.c_dim))
        #plt.imshow(gen)
        #plt.show()
        Image.fromarray(gen).save('{}/{}.jpg'.format(drc, i))
    ims.append(im2)
    ims = np.asarray(ims).reshape((n+2,64,64,3))
    tl.visualize.save_images(ims, [3, (n+2)//3], '{}/vis.png'.format(drc))

if __name__=='__main__':
    #gen()
    interpolate_rand(8)
    #recon()
    #interpolate('interp_select/008247.jpg', 'interp_select/001234.jpg')
    
    
    
'''     
def recon_(file):
    #ds, _ = get_celebA(flags.output_size, 1, 1)
    im = Image.open(file).resize((64, 64))
    plt.imshow(im)
    plt.show() 
    im = np.array(im, dtype=np.float32).reshape((1,64,64,3))
    im = im / 127.5 - 1
    
    rec = G(E(im))
    plt.imshow(np.array((rec + 1.) * 127.5, dtype=np.uint8)[0])
    plt.show()
'''  
'''        
# get optimal encode
def opt(file):
    im = Image.open(file).resize((64, 64))
    plt.imshow(im)
    plt.show() 
    im = np.array(im, dtype=np.float32).reshape((1,64,64,3))
    im = im / 127.5 - 1
    
    ni = tl.layers.Input([1,100])
    nn = tl.layers.Dense(100, act=tf.identity)(ni)
    M = tl.models.Model(inputs=ni, outputs=nn)
    M.train()
    optimizer = tf.optimizers.Adam(0.01)
    z0 = np.random.normal(loc=0., scale=1., size=[1,100]).astype(np.float32)
    for i in range(1000):
        with tf.GradientTape() as tape:
            z_ = M(z0)
            gen = G(z_)
            loss = tl.cost.absolute_difference_error(gen, im, is_mean=True)
        grad = tape.gradient(loss, M.trainable_weights)
        optimizer.apply_gradients(zip(grad, M.trainable_weights))
        print(loss)
    z_ = M(z0)
    gen = G(z_)
    gen = np.array((gen+1.)*127.5, dtype=np.uint8).reshape((64,64,3))
    plt.imshow(gen)
    plt.show()
'''