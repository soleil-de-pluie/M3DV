import data
import os
import os.path
import csv
import keras

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential, Model
from keras.layers import Flatten, Conv3D, MaxPooling3D, MaxPooling2D
from keras.optimizers import Adam, SGD
from matplotlib import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import keras.backend.tensorflow_backend as KTF
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution3D, Convolution2D
from keras.layers.pooling import AveragePooling3D, GlobalAveragePooling3D
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.utils import np_utils

"""
This DenseNet implementation comes from:
    https://github.com/titu1994/DenseNet/blob/master/densenet_fast.py
I've made some modifications so as to make it consistent with Keras2 interface
"""
__all__ = ['create_dense_net']


def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 3x3, Conv2D, optional dropout
    Args:
        input: Input keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added
    '''

    x = Activation('relu')(input)
    x = Convolution3D(nb_filter, (3, 3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional dropout and Maxpooling2D
    Args:
        input: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Convolution3D(nb_filter, (1, 1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(input)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter


def createDenseNet(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                   weight_decay=1E-4, verbose=True):
    ''' Build the create_dense_net model
    Args:
        nb_classes: number of classes
        img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    model_input = Input(shape=img_dim)

    concat_axis = 1 if K.image_data_format() == "channels_first" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution3D(nb_filter, (3, 3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv3D",
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(model_input)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(
        x)

    densenet = Model(inputs=model_input, outputs=x)

    if verbose:
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet


early_stopping = EarlyStopping(monitor='val_loss', patience=40, verbose=2)
checkpointer = ModelCheckpoint(filepath='tmp/%s/weights.{epoch:02d}.h5' % 'newmask5', verbose=1,
                               period=1, save_weights_only=False)

rate = 30

model = createDenseNet(nb_classes=2, img_dim=[32, 32, 32, 1], depth=25, growth_rate=rate)
model.compile(loss=binary_crossentropy, optimizer=SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
#use momentum to accelerate convergence
# print the model
model.summary()

x_path = '/content/gdrive/My Drive/Colab Notebooks/DenseSharp1/data/train_val'
x_test_path = '/content/gdrive/My Drive/Colab Notebooks/DenseSharp1/data/test'


def get_dataset():
    x_return_train = []
    x_name = pd.read_csv("train_val.csv")['name']
    for i in range(465):
        x_file_temp = os.path.join(x_path, x_name[i] + '.npz')
        x_voxel = np.array(np.load(x_file_temp)['voxel'])
        x_seg = np.array(np.load(x_file_temp)['seg'])
        x_temp = x_voxel * x_seg * 0.9 + x_voxel * 0.1
        x_return_train.append(x_temp[34:66, 34:66, 34:66])
    return x_return_train


def get_label():
    x_label = pd.read_csv("train_val.csv")['label']
    x_train_label = keras.utils.to_categorical(x_label, 2)[0:465]
    return x_train_label


def get_testdataset():
    x_return_test = []
    x_name = pd.read_csv("submit.csv")['Id']
    for i in range(117):
        x_file_temp = os.path.join(x_test_path, x_name[i] + '.npz')
        x_voxel = np.array(np.load(x_file_temp)['voxel'])
        x_seg = np.array(np.load(x_file_temp)['seg'])
        x_temp = x_voxel * x_seg * 0.9 + x_voxel * 0.1
        x_return_test.append(x_temp[34:66, 34:66, 34:66])
    return x_return_test


def mixup_data(x1, y1, alpha, n):
    x2 = np.zeros(np.shape(x1))
    y2 = np.zeros(np.shape(y1), 'float')
    l = len(x1)
    ##random array low=0,high=l
    indexs = np.random.randint(0, l, n)
    indexs2 = np.random.randint(0, l, n)
    for i in range(n):
        x2[i] = alpha * x1[indexs2[i]] + (1 - alpha) * x1[indexs[i]]
        y2[i] = alpha * y1[indexs2[i]] + (1 - alpha) * y1[indexs[i]]

    xx = x2[:n]
    yy = y2[:n]
    return xx, yy

def mixup_data1(x1, y1, alpha):
    """Mix data
    x1: input numpy array.
    y1: target numpy array.
    alpha: float.
    """
    x2=np.zeros(np.shape(x1))
    y2=np.zeros(np.shape(y1),'float')
    n = len(x1)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, n)
    else:
        lam = np.array([1.0] * n)
    indexs = np.random.randint(0, n, n)
    for i in range(n):
        x2[i] = x1[i]*lam[i]+(1-lam[i])*x1[indexs[i]]
        y2[i] = y1[i]*lam[i]+(1-lam[i])*y1[indexs[i]]
    return x2, y2

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

# input
x_train = get_dataset()
x_train = np.array(x_train)
x_train_label = get_label()

# mixup
x_train_mixup, x_train_label_mixup = mixup_data(x_train, x_train_label, alpha=0.1, n=50)
x_train = np.r_[x_train, x_train_mixup]
x_train_label = np.r_[x_train_label, x_train_label_mixup]

# reshape
x_train = x_train.reshape(x_train.shape[0], 32, 32, 32, 1)
x_train = x_train.astype('float32') / 255


history = model.fit(x_train, x_train_label, batch_size=32, epochs=150, validation_split=0.2, verbose=2, shuffle=False,
                    callbacks=[early_stopping, checkpointer])
loss, accuracy = model.evaluate(x_train, x_train_label)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (loss, accuracy))

np.savetxt('loss.txt', history.history['loss'])
np.savetxt('val_loss.txt', history.history['val_loss'])
np.savetxt('acc.txt', history.history['acc'])
np.savetxt('val_acc.txt', history.history['val_acc'])

###plot
'''
plt.plot(np.loadtxt('acc.txt'), color='blue', label='acc')
plt.plot(np.loadtxt('val_acc.txt'), color='green', label='val_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(np.loadtxt('loss.txt'), color='blue', label='loss')
plt.plot(np.loadtxt('val_loss.txt'), color='green', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

###test
'''
result = model.predict(y_voxel,batch_size = 1)
np.save("E:/DenseSharp2/tmp/resultfor66/result.npy" , result[0])
result = np.load("E:/DenseSharp2/tmp/resultforfifth/result.npy" )
print(result)
csv = pd.read_csv("E:/DenseSharp1/data/test.csv")
csv.iloc[:, 1] = result[:, 1]
csv.columns = ['Id', 'Predicted']
csv.to_csv("E:/DenseSharp2/tmp/resultfornewmask4/submit125.csv", index=None)
'''