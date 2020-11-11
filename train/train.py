"""
[INTEL CONFIDENTIAL]

Copyright (c) 2020 Intel Corporation.

This software and the related documents are Intel copyrighted materials, and
your use of them is governed by the express license under which they were 
provided to you ("License"). Unless the License provides otherwise, you may
not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express
or implied warranties, other than those that are expressly stated in the License.
"""

import os
import sys
import cv2
import yaml
import math
import datetime as dt

import numpy as np
import pandas as pd
from albumentations import *

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.utils import Sequence
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from utils.yaml_dict import YAMLDict as ydict
from utils.model import build_model

# Read config file
cfg = None
with open('config.yaml', 'r') as stream:
    try:
        cfg = ydict(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(1)
cfg = cfg.base_model_configs

# Set visible GPU devices
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu


def read_csv():
    """ Read train & val data CSV files

        Output
                train_imgs : array of train image file paths
                train_labels : array of train image labels
                val_imgs : array of val image file paths
                val_labels : array of val image labels
    """
    # Read train and val CSV files
    df_train = pd.read_csv(cfg.train_csv_path)
    df_val = pd.read_csv(cfg.val_csv_path)
    # Extract image file paths
    train_imgs = df_train['images']
    val_imgs = df_val['images']
    # Extract labels
    train_labels = df_train['label']
    val_labels = df_val['label']
    return train_imgs, train_labels, val_imgs, val_labels


def lr_decay(epoch):
    """ Learning rate decay function

        Drop learning rate by [cfg.lr_sched_drop] every
        [cfg.epochs_drop] number of epochs

        Input
                epoch : Current epoch count
        Output
                lrate : New learning rate
    """
    initial_lrate = cfg.lr
    drop_rate = cfg.lr_sched_drop
    epochs_until_drop = cfg.epochs_drop
    lrate = initial_lrate * math.pow(drop_rate,
                                     math.floor((1 + epoch) / epochs_until_drop))
    return lrate


def mixup(x, y, alpha=0.3, u=0.5):
    """ Mixup data augmentation

        Mixup helps train the CNN model on pairs of classes and their labels.
        Model is training on a mix of images (linear combination of images)
        in the training set.

        Input
                x : normalized, resized raw image batch
                y : labels associated with the images
        Output
                Returns a linear combinations of the input data
    """
    np.random.seed(42)
    if np.random.random() < u:
        batch_size = len(x)
        lam = np.random.beta(alpha, alpha)
        index = np.random.permutation(batch_size)
        x = lam * x + (1 - lam) * x[index, :]
        y = lam * y + (1 - lam) * y[index, :]
    return x, y


def strong_aug(p=.5):
    """ Composes data augmentation

        Augmentation are based on albumentations library

        Input
                p : Probability of augmentation policy occuring
        Output
                Returns data augmentations to apply to image data
    """
    return Compose([
        Rotate(p=p, border_mode=0),
        Flip(),
        OneOf([
            CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1),
        ], p=1),
    ], p=p)


global global_aug
global_aug = strong_aug(p=cfg.aug_prob)


def freeze_session(session, keep_var_names=None,
                   output_names=None, remove_devices=True):
    """ Freeze graph
    """

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in
                                    tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if remove_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session,
                                                      input_graph_def,
                                                      output_names,
                                                      freeze_var_names)
        return frozen_graph


class DataGenerator(Sequence):
    """ Generates data for training and validation
    """
    def __init__(self, list_IDs, labels, batch_size,
                 n_classes, dim, preprocess_input,
                 shuffle=True, mixupT=True, augT=True):
        """ Initialization
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.dim = dim
        self.preprocess_input = preprocess_input
        self.shuffle = shuffle
        self.mixupT = mixupT
        self.augT = augT

        self.on_epoch_end()

    def __len__(self):
        """ Returns number of batches in dataset
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """ Returns one batch of data
        """
        # Generate start and end indices of the batch
        idxmin = index * self.batch_size
        idxmax = min((index + 1) * self.batch_size, len(self.list_IDs))
        indexes = self.indexes[idxmin:idxmax]
        # Find IDs and labels for the selected indices
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_labels_temp = [self.labels[k] for k in indexes]
        # Generate data for selected IDs and labels
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp)
        return X, y

    def on_epoch_end(self):
        """ Updates indexes after each epoch
        """
        # Shuffle array of indices after each epoch end
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(42)
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_labels_temp):
        """ Generates data containing batch_size samples
        """
        y_batch = []
        x_batch = []
        for i, ID in enumerate(list_IDs_temp):
            # Read and resize images
            image = cv2.imread(ID, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(list_labels_temp[i], cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.resize(image, self.dim)
            label = cv2.resize(label, self.dim)

            # Apply augmentations to images
            if self.augT == True:
                data = {"image": image, "mask": label}
                augmented = global_aug(**data)
                image, label = augmented["image"], augmented["mask"]
                # cv2.imwrite(os.path.join('/home/dls1/Desktop/gen', os.path.basename(ID)), image)
            # Append images to batch array
            label_map = np.zeros((self.dim[1], self.dim[0], self.n_classes), dtype=np.float)
            label_map[label > 0] = 1.0
            x_batch.append(image)
            y_batch.append(label_map)
        # Normalize image array
        x_batch = np.array(x_batch)
        X = preprocess_input(x_batch)
        if self.mixupT == True:
            # Include mixup augmentation policies
            x_batch, y_batch = mixup(x=X,
                                     y=np.array(list_labels_temp))
        else:
            x_batch = X
            y_batch = np.array(y_batch)
        return x_batch, y_batch


if __name__ == '__main__':
    """ Main function

        Train a binary classification model
        Save checkpoint weights ./weights/
        Save Tensorboard files ./tf_logs/
        Save training log files in ./logs/
        Save Model file in ./models/
    """
    # Create directory to store weights and models
    if not os.path.exists('weights'):
        os.mkdir('./weights/')
    if not os.path.exists('models'):
        os.mkdir('./models/')
    if not os.path.exists('logs'):
        os.mkdir('./logs/')

    # Get data
    train_imgs, train_labels, \
    val_imgs, val_labels = read_csv()

    # Print size of all image/label lists
    print("Train imgs, labels : {}, {}".format(len(train_imgs), len(train_labels)))
    print("Val imgs, labels : {}, {}".format(len(val_imgs), len(val_labels)))

    # Build binary CNN model
    model, preprocess_input = build_model(cfg)

    # Store training start time
    train_start_time = dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # Define checkpoint file path
    chkpt_path = './weights/bin_' + cfg.base_model + \
                 '_' + train_start_time + \
                 '_ep:{epoch:02d}_vloss:{val_loss:.5f}.h5'

    # Callbacks
    callbacks = [
        LearningRateScheduler(lr_decay),
        EarlyStopping(monitor='loss',
                      patience=50,
                      verbose=1,
                      min_delta=1e-5),
        ReduceLROnPlateau(monitor='loss',
                          factor=0.1,
                          patience=50,
                          verbose=1,
                          min_delta=1e-5),
        ModelCheckpoint(filepath=chkpt_path,
                        monitor='loss',
                        mode='auto',
                        save_best_only=True,
                        save_weights_only=False),
        TensorBoard(log_dir='tf_logs'),
        CSVLogger('./logs/bin_{}_{}.csv'.format(cfg.base_model,
                                                train_start_time))
    ]
    # Initialize train data generator
    training_generator = DataGenerator(train_imgs,
                                       labels=train_labels,
                                       batch_size=cfg.batch_size,
                                       n_classes=1,
                                       dim=(cfg.img_width, cfg.img_height),
                                       preprocess_input=preprocess_input,
                                       shuffle=True,
                                       mixupT=False,
                                       augT=True)

    # Initialize validation generator
    validation_generator = DataGenerator(val_imgs,
                                         labels=val_labels,
                                         batch_size=cfg.batch_size,
                                         n_classes=1,
                                         dim=(cfg.img_width, cfg.img_height),
                                         preprocess_input=preprocess_input,
                                         shuffle=False,
                                         mixupT=False,
                                         augT=False)

    # Train model
    model.fit_generator(generator=training_generator,
                        epochs=cfg.epochs,
                        verbose=1,
                        callbacks=callbacks,
                        initial_epoch=0,
                        workers=cfg.max_workers,
                        max_queue_size=12,
                        use_multiprocessing=cfg.multi_proc,
                        validation_data=validation_generator)

    # Freeze graph
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])

    # Save trained model
    print('')
    print('******* Saving CNN binary segmentation model *******')
    print('./models/bin_{}_{}.hdf5'.format(cfg.base_model, train_start_time))
    print('./models/bin_{}_{}.pb'.format(cfg.base_model, train_start_time))
    print('')
    # Saving .hdf5 model file
    model.save('./models/bin_{}_{}.hdf5'.format(cfg.base_model,
                                                train_start_time))
    # Saving .pb model file
    tf.train.write_graph(frozen_graph,
                         './models/',
                         'bin_{}_{}.pb'.format(cfg.base_model, train_start_time),
                         as_text=False)
