# imports
from pathlib import Path
import numpy as np
from scipy.signal import resample
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, BatchNormalization, MaxPooling1D
from keras import regularizers
from keras import initializers
from keras.optimizers import Adam, SGD, Adagrad
from keras.utils import np_utils
from keras.callbacks import TensorBoard, CSVLogger

from experiment_utils import *
from report_generation import *


## one time parameter for the model below
## regularizer
## l2
ker_reg = 0.1
act_reg = 0.1
## kernel_initializer
ker_init = initializers.glorot_normal(seed=None)
## shape
in_shape = (648, 4)
## learning rate
opt = Adam()
opt.lr = 0.0001
##
OUTPUT_SIZE = 2
## batch size
bsize = 50
##
epochs = 30
## callback
model_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

def make_model():
    ## model
    ## resample data to 648 * 1
    model = Sequential()
    ## 1d conv, size 3 filter, 64 filters, stride 1
    ## batch norm, batch after activation
    ## no maxpool
    ## keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    model.add(Convolution1D(filters=64, kernel_size=3, strides=1, padding='same', input_shape=in_shape, kernel_initializer=ker_init, activation='relu', kernel_regularizer=regularizers.l2(ker_reg)))
    model.add(BatchNormalization())
    ## 1d conv, size 3 filter, 128 filters, stride 1
    ## batch norm, batch after activation
    ## maxpool 3 --> 216 * 128
    model.add(Convolution1D(filters=128, kernel_size=3, strides=1, padding='same', input_shape=in_shape, activation='relu', kernel_regularizer=regularizers.l2(ker_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3))
    ## 1d conv, size 3 filter, 128 filters, stride 2
    ## batch norm, batch after activation
    ## max pool 3 -->  36 * 128
    model.add(Convolution1D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(ker_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3))
    ## 1d conv, size 3 filter, 256 filters, stride 2
    ## batch norm, batch after activation
    ## max pool 3 -->  6 * 256
    model.add(Convolution1D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(ker_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3))
    ## 1d conv, size 3 filter, 512 filters, stride 2
    ## batch norm, batch after activation
    ## max pool 3 -->  1 * 512
    model.add(Convolution1D(filters=512, kernel_size=3, strides=2, padding='same',activation='relu', kernel_regularizer=regularizers.l2(ker_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3))
    ##
    model.add(Flatten())
    ## fully connected
    model.add(Dense(OUTPUT_SIZE))
    ## softmax
    model.add(Activation('softmax'))

    return model

def experiment(model):
    ## experiment
    ## case 12, 14 for test
    ## case 18 for validation
    ## other case for traning
    x_list, y_list = load_waveforms()
    x_list = positve_samples(x_list)
    x_list = split_by_channel(x_list)
    x_list = apply_resample(x_list, 648)
    y_list = binary_label(y_list)
    for i in range(x_list.shape[0]):
        print(x_list[i].shape)

    # val_idx = [17]
    # test_idx = [11, 13]
    ## run iterate_samples
    ## use test set as validation, to see the loss change for each fold
    indices = iterate_samples(20, 1, 2, 1)
    for idx in indices:
        val_idx = idx[2]
        test_idx = [] ## no test set

        train_list_x = []
        train_list_y = []
        val_list_x = []
        val_list_y = []
        test_list_x = []
        test_list_y = []
        for idx in range(x_list.shape[0]):
            if idx not in (val_idx + test_idx):
                train_list_x.append(x_list[idx])
                train_list_y.append(y_list[idx])
                
        for idx in val_idx:
            val_list_x.append(x_list[idx])
            val_list_y.append(y_list[idx])
            
        for idx in test_idx:
            test_list_x.append(x_list[idx])
            test_list_y.append(y_list[idx])  

        train_list_x = np.array(train_list_x)
        train_list_y = np.array(train_list_y)
        val_list_x = np.array(val_list_x)
        val_list_y = np.array(val_list_y)
        test_list_x = np.array(test_list_x)
        test_list_y = np.array(test_list_y)
        train_list_x = combine_samples(train_list_x)
        train_list_y = combine_samples(train_list_y)
        val_list_x = combine_samples(val_list_x)
        val_list_y = combine_samples(val_list_y)
        test_list_x = combine_samples(test_list_x)
        test_list_y = combine_samples(test_list_y)
        train_list_y = np_utils.to_categorical(train_list_y, num_classes=2)
        val_list_y = np_utils.to_categorical(val_list_y, num_classes=2)
        test_list_y = np_utils.to_categorical(test_list_y, num_classes=2)

        model.fit(train_list_x, train_list_y,
                epochs=epochs,
                batch_size=bsize
                verbose=2,
                validation_data=(test_list_x, test_list_y),
                callbacks=[model_callback])

def main():
    model = make_model()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    experiment(model)

if __name__ == '__main__':
    main()