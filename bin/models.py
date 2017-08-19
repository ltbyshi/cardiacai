import os
import keras
from keras.models import Sequential, Model
from keras.engine.topology import Input, InputLayer, Layer
from keras.layers.core import RepeatVector, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv1D, Conv2D, UpSampling2D
from keras.layers import Activation, Dense
from keras.layers.pooling import AveragePooling1D, MaxPooling1D, MaxPooling2D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l1, l2, l1_l2
from keras.layers.merge import Add, Concatenate, Multiply
from keras.optimizers import SGD, RMSprop, Adam
from keras.losses import binary_crossentropy
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import keras.backend.tensorflow_backend

def _get_session():
    """Modified the original get_session() function to change the ConfigProto variable
    """
    global _SESSION
    if tf.get_default_session() is not None:
        session = tf.get_default_session()
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                        allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if not _MANUAL_VAR_INIT:
        _initialize_variables()
    return session

# control GPU memory usage for TensorFlow backend
if K.backend() == 'tensorflow':
    # replace the original get_session() function
    keras.backend.tensorflow_backend.get_session.func_code = _get_session.func_code
    import tensorflow as tf

def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return (2.0*K.sum(y_true*y_pred) + 1.0)/(K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + 1.0)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)



# custom objects that can be passed to the keras.models.load_model function
custom_objects = {'dice_coef': dice_coef,
                   'dice_coef_loss': dice_coef_loss}

def vgg16(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape, name='conv1_1'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_2'))
    model.add(MaxPooling2D(2, 2, name='pool1'))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_1'))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_2'))
    model.add(MaxPooling2D(2, 2, name='pool2'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_1'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_2'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_3'))
    model.add(MaxPooling2D(2, 2, name='pool3'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_1'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_2'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_3'))
    model.add(MaxPooling2D(2, 2, name='pool4'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_1'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_2'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_3'))
    model.add(MaxPooling2D(2, 2, name='pool5'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name='fc8'))
    optimizer = RMSprop(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model

def unet1(input_shape):
    """
    Build a U-net for image segmentation
    Refer to: https://github.com/yihui-he/u-net/blob/master/train.py.
    :param input_shape: (nrow, ncol)
    :return: a keras model
    """
    input = Input(shape=input_shape)
    down1_conv1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name='down1_conv1')(input)
    down1_conv2 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name='down1_conv2')(down1_conv1)
    down1_pool1 = MaxPooling2D(pool_size=(2, 2), name='down1_pool1')(down1_conv2)

    down2_conv1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='down2_conv1')(down1_pool1)
    down2_conv2 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='down2_conv2')(down2_conv1)
    down2_pool1 = MaxPooling2D(pool_size=(2, 2), name='down2_pool1')(down2_conv2)

    down3_conv1 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='down3_conv1')(down2_pool1)
    down3_conv2 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='down3_conv2')(down3_conv1)
    down3_pool1 = MaxPooling2D(pool_size=(2, 2), name='down3_pool1')(down3_conv2)

    down4_conv1 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='down4_conv1')(down3_pool1)
    down4_conv2 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='down4_conv2')(down4_conv1)
    down4_pool1 = MaxPooling2D(pool_size=(2, 2), name='down4_pool1')(down4_conv2)

    down5_conv1 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='down5_conv1')(down4_pool1)
    down5_conv2 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='down5_conv2')(down5_conv1)

    up4_upsample = UpSampling2D(size=(2, 2), name='up4_upsample')(down5_conv2)
    up4_upconv = Conv2D(256, kernel_size=(2, 2), padding='same', activation='relu',  name='up4_upconv')(up4_upsample)
    up4_merge = Concatenate(axis=-1, name='up4_merge')([up4_upconv, down4_conv2])
    up4_conv1 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='up4_conv1')(up4_merge)
    up4_conv2 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='up4_conv2')(up4_conv1)

    up3_upsample = UpSampling2D(size=(2, 2), name='up3_upsample')(up4_conv2)
    up3_upconv = Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu', name='up3_upconv')(up3_upsample)
    up3_merge = Concatenate(axis=-1, name='up3_merge')([up3_upconv, down3_conv2])
    up3_conv1 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='up3_conv1')(up3_merge)
    up3_conv2 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='up3_conv2')(up3_conv1)

    up2_upsample = UpSampling2D(size=(2, 2), name='up2_upsample')(up3_conv2)
    up2_upconv = Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu', name='up2_upconv')(up2_upsample)
    up2_merge = Concatenate(axis=-1, name='up2_merge')([up2_upconv, down2_conv2])
    up2_conv1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='up2_conv1')(up2_merge)
    up2_conv2 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='up2_conv2')(up2_conv1)

    up1_upsample = UpSampling2D(size=(2, 2), name='up1_upsample')(up2_conv2)
    up1_upconv = Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu', name='up1_upconv')(up1_upsample)
    up1_merge = Concatenate(axis=-1, name='up1_merge')([up1_upconv, down1_conv2])
    up1_conv1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name='up1_conv1')(up1_merge)
    up1_conv2 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name='up1_conv2')(up1_conv1)

    predict = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(up1_conv2)

    model = Model(inputs=[input], outputs=[predict])



    #optimizer = SGD(lr=0.01, momentum=0.9)
    optimizer = Adam(lr=0.001)
    loss = dice_coef_loss
    metrics = [dice_coef]
    loss = 'binary_crossentropy'
    metrics = ['binary_accuracy']
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model

def unet_from_vgg16(model, fine_tune=True):
    if fine_tune:
        for layer in model.layers:
            layer.trainable = False
    up_block5_upsample = UpSampling2D(size=(2, 2), name='up_block5_upsample')(model.get_layer('block5_conv3').output)
    up_block5_upconv = Conv2D(512, kernel_size=(2, 2), padding='same', activation='relu', name='up_block5_upconv')(up_block5_upsample)
    up_block5_merge = Concatenate(axis=-1, name='up_block5_merge')([up_block5_upconv, model.get_layer('block4_conv3').output])
    up_block5_conv1 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='up_block5_conv1')(up_block5_merge)
    up_block5_conv2 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='up_block5_conv2')(up_block5_conv1)
    up_block5_conv3 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='up_block5_conv3')(up_block5_conv2)

    up_block4_upsample = UpSampling2D(size=(2, 2), name='up_block4_upsample')(up_block5_conv3)
    up_block4_upconv = Conv2D(256, kernel_size=(2, 2), padding='same', activation='relu', name='up_block4_upconv')(up_block4_upsample)
    up_block4_merge = Concatenate(axis=-1, name='up_block4_merge')([up_block4_upconv, model.get_layer('block3_conv3').output])
    up_block4_conv1 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='up_block4_conv1')(up_block4_merge)
    up_block4_conv2 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='up_block4_conv2')(up_block4_conv1)
    up_block4_conv3 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='up_block4_conv3')(up_block4_conv2)

    up_block3_upsample = UpSampling2D(size=(2, 2), name='up_block3_upsample')(up_block4_conv3)
    up_block3_upconv = Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu', name='up_block3_upconv')(up_block3_upsample)
    up_block3_merge = Concatenate(axis=-1, name='up_block3_merge')([up_block3_upconv, model.get_layer('block2_conv2').output])
    up_block3_conv1 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='up_block3_conv1')(up_block3_merge)
    up_block3_conv2 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='up_block3_conv2')(up_block3_conv1)

    up_block2_upsample = UpSampling2D(size=(2, 2), name='up_block2_upsample')(up_block3_conv2)
    up_block2_upconv = Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu', name='up_block2_upconv')(up_block2_upsample)
    up_block2_merge = Concatenate(axis=-1, name='up_block2_merge')([up_block2_upconv, model.get_layer('block1_conv2').output])
    up_block2_conv1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='up_block2_conv1')(up_block2_merge)
    up_block2_conv2 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='up_block2_conv2')(up_block2_conv1)

    predict = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', name='predict')(up_block2_conv2)

    unet_model = Model(inputs=[model.input], outputs=[predict], name='unet_vgg16')

    optimizer = Adam(lr=0.001)
    loss = 'binary_crossentropy'
    metrics = ['binary_accuracy']
    unet_model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return unet_model

def get_pretrained_vgg16(filename, input_shape):
    model = vgg16(input_shape)
    import numpy as np
    weights = np.load(filename)[()]
    weight_tuples = []
    model_weights = {v.name:v for v in model.trainable_weights}
    for layer_name in weights.keys():
        if layer_name in ['fc6', 'fc7', 'fc8']:
            continue
        if layer_name not in ['fc6', 'fc7', 'fc8']:
            model.get_layer(layer_name).trainable = False
        weight_tuples.append((model_weights[layer_name + '/kernel:0'], weights[layer_name]['weights']))
        weight_tuples.append((model_weights[layer_name + '/bias:0'], weights[layer_name]['biases']))
    K.batch_set_value(weight_tuples)
    return model

def get_model(name, input_shape):
    if name == 'vgg16':
        return vgg16(input_shape)

def add_fc_layers(model, n_classes=2):
    """
    Add fully-connected layers to a pretrained model for classification
    """
    flatten = Flatten(name='flatten')(model.output)
    fc1 = Dense(1024, activation='relu', name='fc1')(flatten)
    #fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(1024, activation='relu', name='fc2')(fc1)
    #fc2 = Dropout(0.5)(fc2)
    if n_classes > 2:
        fc3 = Dense(n_classes, activation='softmax', name='fc3')(fc2)
        loss = 'categorical_crossentropy'
        metrics = ['categorical_accuracy']
    else:
        fc3 = Dense(1, activation='sigmoid', name='fc3')(fc2)
        loss = 'binary_crossentropy'
        metrics = ['binary_accuracy']
    model = Model(inputs=model.input, outputs=fc3)
    optimizer = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss=loss, metrics=metrics)
    return model

"""Custom keras layers
"""

def pool2d_with_argmax(x, pool_size, strides=(1, 1),
           padding='valid', data_format=None,
           pool_mode='max'):
    """2D Pooling.

    # Arguments
        x: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        pool_mode: string, `"max"` or `"avg"`.

    # Returns
        A tensor, result of 2D pooling.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    padding = _preprocess_padding(padding)
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)

    x = _preprocess_conv2d_input(x, data_format)

    if pool_mode == 'max':
        x, argmax = tf.nn.max_pool_with_argmax(x, pool_size, strides, padding=padding)
    else:
        raise ValueError('Invalid pooling mode:', pool_mode)

    return _postprocess_conv2d_output(x, data_format), argmax
keras.backend.tensorflow_backend.pool2d_with_argmax = keras.backend.tensorflow_backend.pool2d
keras.backend.tensorflow_backend.pool2d_with_argmax.func_code = keras.backend.tensorflow_backend.pool2d.func_code

class MaxPooling2DWithArgMax(MaxPooling2D):
    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output, argmax = K.pool2d_with_argmax(inputs, pool_size, strides,
                          padding, data_format,
                          pool_mode='max')
        self.argmax = argmax
        return output
keras.layers.pooling.MaxPooling2DWithArgMax = MaxPooling2DWithArgMax

def get_deconvnet_vgg16(filename):
    import h5py
    import json
    from keras.models import model_from_config
    with h5py.File(filename, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
        model_config = json.loads(model_config.decode('utf-8'))
        for layer in model_config['config']['layers']:
            if layer['class_name'] == u'MaxPooling2D':
                layer['class_name'] = u'MaxPooling2DWithArgMax'
        model = model_from_config(model_config, custom_objects=custom_objects)
    f.close()
    model.load_weights(filename)
    model.get_layer('block1_pool1')