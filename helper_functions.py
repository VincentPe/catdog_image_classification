import os
import cv2
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import shutil

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import *
from keras.callbacks import Callback
from tensorflow.keras import backend as K
from tqdm import tqdm


# UDFs

def initialize_simplified_VGG(rows: int, cols: int, channels: int):
    """
    Initialize Sequential model and add layers in smaller form of VGG

    :param rows: Height of input pictures in nr of pixels
    :param cols: Width of input pictures in nr of pixels
    :param channels: Third dimension of input image, i.e. colors
    :return: Sequential model
    """
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu, padding='same', 
                               input_shape=(rows, cols, channels)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),

        tf.keras.layers.Dense(2, activation='softmax')
    ])

    return model


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular": A basic triangular cycle w/ no amplitude scaling.
    "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range": A cycle that scales initial amplitude by gamma**(cycle iterations) at each cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000., scale_fn=clr_fn, scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally, it defines the cycle amplitude (max_lr - base_lr).
                The lr at any cycle is the sum of base_lr and some scaling of the amplitude; 
                therefore max_lr may not actually be reached depending on scaling function.
        step_size: number of training iterations per half cycle.
                   Authors suggest setting step_size 2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
              Default 'triangular'. Values correspond to policies detailed above.
              If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function: gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single argument lambda function, 
                  where 0 <= scale_fn(x) <= 1 for all x >= 0. mode parameter is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on cycle number or cycle iterations
            (training iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular', gamma=1., scale_fn=None,
                 scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """Resets cycle iterations. Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


def read_image(file_path, rows, cols):
    """Reads image from file path and immediately resizes"""
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) 
    return cv2.resize(img, (rows, cols), interpolation=cv2.INTER_CUBIC)


def prep_data(images, rows, cols, channels):
    """Creates array from preprocessed images.
    Shows progress bar while doing so"""
    count = len(images)
    data = np.ndarray((count, rows, cols, channels), dtype=np.uint8)

    for i, image_file in enumerate(tqdm(images)):
        image = read_image(image_file, rows, cols)
        data[i] = image
    
    return data
    
    
def sort_test_images(test_dir):
    """Orders the test images in right order to comply with Kaggle's expected submission format"""
    for i in os.listdir(test_dir):
        os.rename(test_dir + i, test_dir + i.split('.')[0].zfill(6) + '.jpg')
    test_images = [test_dir+i for i in os.listdir(test_dir)]
    test_images.sort()


def preprocess_images(train_dir, test_dir, test_images, rows, cols, channels, sample=False, test_size=0.1):
    """Does all the required preprocessing for both train and test images.
    Has possibility to just use a sample of data"""

    train_dogs = [train_dir+i for i in os.listdir(train_dir) if 'dog' in i]
    train_cats = [train_dir+i for i in os.listdir(train_dir) if 'cat' in i]

    sort_test_images(test_dir)

    if not sample:
        # Unsampled
        X_files = train_dogs + train_cats
        y = np.concatenate([np.repeat(1, 12500), np.repeat(0, 12500)])

    else:
        # sample
        n_samples = sample * len(train_dogs)
        X_files = train_dogs[:n_samples] + train_cats[:n_samples]
        y = np.concatenate([np.repeat(1, 1250), np.repeat(0, 1250)])

    X_train_files, X_test_files, y_train, y_test = train_test_split(X_files, y, test_size=test_size, random_state=0)

    X_train = prep_data(X_train_files, rows, cols, channels)
    X_test = prep_data(X_test_files, rows, cols, channels)
    comp_data = prep_data(test_images, rows, cols, channels)

    # Normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    comp_data = comp_data / 255.0

    return comp_data, X_train, X_test, y_train, y_test


def show_RAM_items():
    """
    Get a sorted list of the objects and their sizes
    :return:
    """
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and
            x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
    
    
def training_history_plots(history, clr):

    # Show cyclical learning rate
    plt.subplot(1, 3, 1)
    plt.xlabel('Training Iterations')
    plt.ylabel('Learning Rate')
    plt.title("CLR - 'triangular' Policy")
    plt.plot(clr.history['iterations'], clr.history['lr'])

    # summarize history for accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    
def show_directory_structure(startpath):
    
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 3 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))


def copy_files_for_imagedatagen(filenames, folder):
    for filename in filenames:
        if 'dog' in filename:
            shutil.copy(filename, folder + '/dogs/' + filename.split('/')[-1])
        else:
            shutil.copy(filename, folder + '/cats/' + filename.split('/')[-1])
            
            
def mkdir_ifnotexist(folder): 
    if not os.path.exists(folder):
        os.mkdir(folder)


def prepare_file_structure(X_train_filenames, X_val_filenames, X_test_filenames):

    mkdir_ifnotexist('data')
    for i in ['train', 'validation', 'test']:
        mkdir_ifnotexist('data/' + i)
        mkdir_ifnotexist('data/' + i + '/dogs')
        mkdir_ifnotexist('data/' + i + '/cats')

    copy_files_for_imagedatagen(X_train_filenames, 'data/train')
    copy_files_for_imagedatagen(X_val_filenames, 'data/validation')
    copy_files_for_imagedatagen(X_test_filenames, 'data/test')
