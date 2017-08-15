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
    optimizer = Adam(lr=0.001)
    #optimizer = RMSprop(lr=0.001)
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
    fc2 = Dense(1024, activation='relu', name='fc2')(fc1)
    if n_classes > 2:
        fc3 = Dense(n_classes, activation='softmax', name='fc3')(fc2)
        loss = 'categorical_crossentropy'
        metrics = ['categorical_accuracy']
    else:
        fc3 = Dense(1, activation='sigmoid', name='fc3')(fc2)
        loss = 'binary_crossentropy'
        metrics = ['binary_accuracy']
    model = Model(inputs=model.input, outputs=fc3)
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9),
                  loss=loss, metrics=metrics)
    return model