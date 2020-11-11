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

import ast
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.utils import Sequence
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def decoder_block(filters, kernel_size, stage):
    """ Build decoder block
        Input
                filters : filters to do concat
                stage : decode stage
        Output
                layer : decoder transpose layer
    """
    # Ops Names
    transp_name = 'decoder{}_transpose'.format(stage)
    bn_name = 'decoder{}_bn'.format(stage)
    relu_name = 'decoder{}_relu'.format(stage)
    conv_name = 'decoder{}_conv'.format(stage)
    concat_name = 'decoder{}_concat'.format(stage)
    axis = 3 if K.image_data_format() == 'channels_last' else 1

    # Build the block
    def layer(input_tensor, skip=None):
        # The strides=(2, 2) and kernel=(4, 4) size are fixed in this decoder layer.
        # This is to get the fixed 2X upsampling factors for rows and columns.
        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=False,
        )(input_tensor)
        x = layers.BatchNormalization(axis=axis, name=bn_name)(x)
        x = layers.Activation('relu', name=relu_name)(x)
        if skip is not None:
            x = layers.Concatenate(axis=axis, name=concat_name)([x, skip])

        # Conv layer
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            use_bias=False,
            name=conv_name,
            kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization(axis=axis, name=None)(x)
        x = layers.Activation('relu', name=None)(x)
        return x

    return layer


def CNN(cfg, basemodel):
    """ Build CNN model

        Input
                input_width : input image width
                input_height : input image height
                decoder_filters : filters to do concat
        Output
                model : CNN model
    """
    # Load base model
    model = basemodel(input_shape=(cfg.img_height, cfg.img_width, 3),
                      weights='imagenet', include_top=False)

    # Feature layers corresponding to decoder filter
    _default_feature_layers = {
        # Vgg
        'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
        'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),

        # MobileNet
        'mobilenet': ('conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'),
        'mobilenetv2': ('block_13_expand_relu', 'block_6_expand_relu', 'block_3_expand_relu',
                        'block_1_expand_relu'),

        # DenseNet
        'densenet121': (311, 139, 51, 4),
        'densenet169': (367, 139, 51, 4),
        'densenet201': (479, 139, 51, 4),

        # ResNet
        'resnet50': ('activation_39', 'activation_21', 'activation_9', 'activation'),
    }

    # Concat layers
    skip_connection_layers = _default_feature_layers[cfg.base_application.lower()]
    input_ = model.input
    x = model.output
    skips = ([model.get_layer(name=i).output if isinstance(i, str)
              else model.get_layer(index=i).output for i in skip_connection_layers])

    decoder_filters = ast.literal_eval(cfg.decoder_filters)
    for i in range(len(decoder_filters)):
        if i < len(skips):
            skip = skips[i]
        else:
            skip = None
        x = decoder_block(
            filters=decoder_filters[i],
            kernel_size=3,
            stage=i)(x, skip)

    # Final Conv layer
    x = layers.Conv2D(
        filters=1,
        kernel_size=3,
        padding='same',
        name='final_conv')(x)
    x = layers.Activation('sigmoid', name='sigmoid')(x)

    model = Model(input_, x)
    return model


def iou_score(y_true, y_pred):
    """ Intersection over Union (IoU) for sematic segmentation

        Input
                y_true : array of ground truth labels for data
                y_pred : array of predicted labels after activation layer
        Output
                IoU : calculated IoU
    """
    smooth = 1e-5
    axes = [1, 2]
    axes.insert(0, 0)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true + y_pred, axis=axes) - intersection

    x = (intersection + smooth) / (union + smooth)
    IoU = K.mean(x)
    return IoU


def f1_score(y_true, y_pred):
    """ F-score for training on imbalanced classes

        Input
                y_true : array of ground truth labels for data
                y_pred : array of predicted labels after activation layer
        Output
                FScore : calculated the balanced F-score
    """
    beta = 1
    smooth = 1e-5
    axes = [1, 2]
    axes.insert(0, 0)
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    x = ((1 + beta ** 2) * tp + smooth) \
        / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    FScore = K.mean(x)
    return FScore


def dice_loss(y_true, y_pred):
    """ Loss function for training on imbalanced classes

        Input
                y_true : array of ground truth labels for data
                y_pred : array of predicted labels after activation layer
        Output
                loss : calculated loss to be used for optimization
    """
    loss = 1 - f1_score(y_true, y_pred)
    return loss


def build_model(cfg):

    # Load tensorflow keras applications path
    kerasapp = str("tensorflow.keras.applications")
    # Import module tfkerasapp
    bmodel = __import__(kerasapp, fromlist=[cfg.base_model])
    # Get the base_model call fnction
    app_model = getattr(bmodel, cfg.base_model)
    # Get base_application model funtion from app_model
    basemodel = getattr(app_model, cfg.base_application)
    # Get preprocess_input funtion from app_model
    preprocess_input = getattr(app_model, "preprocess_input")

    # Build binary CNN model
    model = CNN(cfg, basemodel)

    # Display model summary
    model.summary(positions=[.6, 1., 1., 1.])

    # Compile model
    if cfg.optimizer == 'Adam':
        model.compile(optimizer=optimizers.Adam(lr=cfg.lr),
                      loss=dice_loss,
                      metrics=['accuracy', iou_score])
    elif cfg.optimizer == 'RMS':
        model.compile(optimizer=optimizers.RMSprop(lr=cfg.lr),
                      loss=dice_loss,
                      metrics=['accuracy', iou_score])
    elif cfg.optimizer == 'SGD':
        model.compile(optimizer=optimizers.SGD(lr=cfg.lr),
                      loss=dice_loss,
                      metrics=['accuracy', iou_score])

    return model, preprocess_input