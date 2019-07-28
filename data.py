# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:44:08 2019

@author: Yhq
"""

import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import multiprocessing

#tl.logging.set_verbosity(tl.logging.DEBUG)

class FLAGS(object):
    def __init__(self):
        self.n_epoch = 25 # Epoch to train
        self.z_dim = 100 # Dim of noise vector
        self.lr = 0.0002 # Learning rate for adam
        self.beta1 = 0.5 # Momentum term of adam
        self.batch_size = 64 # Batch size
        self.output_size = 64 # Size of the output images
        self.sample_size = 64 # The number of sample images
        self.c_dim = 3 # Number of image channels
        self.save_every_epoch = 1 # Interval of saving checkpoints
        self.print_every_step = 10 # Interval of print training info
        self.dataset = "data" # Dataset dir
        self.checkpoint_dir = "checkpoint" # Checkpoint dir
        self.sample_dir = "samples" # Samples dir
        assert np.sqrt(self.sample_size) % 1 == 0.
        
flags = FLAGS() 

tl.files.exists_or_mkdir(flags.checkpoint_dir) # save model
tl.files.exists_or_mkdir(flags.sample_dir) # save generated image

def get_celebA(output_size, n_epoch, batch_size):
    # dataset API and augmentation
    images_path = tl.files.load_file_list(path=flags.dataset, regx='.*.jpg', keep_prefix=True, printable=False)
    def generator_train():
        for image_path in images_path:
            yield image_path.encode('utf-8')
    def _map_fn(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = image[45:173, 25:153, :] # central crop
        image = tf.image.resize([image], (output_size, output_size))[0]
        image = tf.image.random_flip_left_right(image)
        image = image * 2 - 1 # RGB in [-1,1]
        return image
    
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.string)
    ds = train_ds.shuffle(buffer_size=4096)
    ds = ds.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count()) # parallel process
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    return ds, images_path
