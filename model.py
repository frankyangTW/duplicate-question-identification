#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Baseline
"""
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
import time
from keras.models import Model, Input, load_model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from metrics import f1, precision, MCC, FPR, TPR, FNR, TNR
from utils import AucCallback, limitGPU, loadDataset,load_embedding_matrix, saveHistory, layer_norm
import tensorflow as tf

limitGPU()

###########################################################
MODEL_NAME = 'base'
###########################################################

# Define constants
REGULARIZE = 0.0001
GRU_REGULARIZE = 0.0005
MAX_LENGTH = 30
DROPOUT = 0.2
HIDDEN_RNN_UNITS = 192
HIDDEN_DENSE_UNITS = 2048
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 64

## Load Datasets
train_x1, train_x2, train_features, train_y, valid_x1, valid_x2, valid_y, valid_features = loadDataset()
print ('Dataset Loaded')

start_time = time.time()

## Load Embedding Matrix
(embedding_matrix, vocab_size) = load_embedding_matrix()

## Define Model
def build_model():
    input_1 = Input(shape=(MAX_LENGTH, ))
    input_2 = Input(shape=(MAX_LENGTH, ))

    e = Embedding(vocab_size, 300,
                  weights=[embedding_matrix], input_length=MAX_LENGTH,
                  trainable=False)
    encoded1 = e(input_1)
    encoded2 = e(input_2)

    sharedRNN = CuDNNGRU(HIDDEN_RNN_UNITS, kernel_regularizer=l2(GRU_REGULARIZE))

    encoded1 = Lambda(layer_norm)(encoded1)
    encoded2 = Lambda(layer_norm)(encoded2)

    encoder1 = sharedRNN(encoded1)
    encoder2 = sharedRNN(encoded2)

    diff = subtract([encoder1, encoder2])
    diff_square = Lambda(lambda x: x ** 2)(diff)
    dot_layer = multiply([encoder1, encoder2])
    merged = concatenate([encoder1, encoder2, diff_square, dot_layer])
    merged = Dropout(DROPOUT)(merged)

    dense_out = BatchNormalization()(merged)
    dense_out = Dense(HIDDEN_DENSE_UNITS,
                      kernel_regularizer=l2(REGULARIZE), activation='relu'
                      )(dense_out)
    dense_out = Dropout(DROPOUT)(dense_out)

    dense_out = Dense(1,
                      kernel_regularizer=l2(REGULARIZE), activation='hard_sigmoid'
                      )(dense_out)

    model = Model(inputs=[input_1, input_2], outputs=[dense_out])
    model.summary()

    adam = Adam(lr=LEARNING_RATE, clipnorm=1.0)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=['accuracy', f1, MCC, FPR, TPR, FNR, TNR])
    return model

## Define Callbacks
auc = AucCallback(([train_x1, train_x2], train_y),
                    ([valid_x1, valid_x2], valid_y))

stop = EarlyStopping(monitor='val_acc', patience=11)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=10)

log_file = MODEL_NAME+'.csv'
csv = CSVLogger(log_file, append=True)

class_weights = {   0: 1.,
                    1: 1}

lr_schedule = LearningRateScheduler(lambda epoch: LEARNING_RATE * (0.2 ** int(epoch/10)))

check_point = ModelCheckpoint('{epoch:02d}_{val_acc:.4f}.h5', monitor='val_acc')

## Train
model = build_model()
history = model.fit(
    [train_x1, train_x2],
    train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=([valid_x1, valid_x2], valid_y),
    callbacks=[auc, stop, csv, reduce_lr, lr_schedule, check_point],
    shuffle=True,
    class_weight=class_weights
    )


# save Model
model.save(MODEL_NAME+'.h5')
print ('model saved')
# save history
HISTORY_FILE = MODEL_NAME+'.pkl'

saveHistory(history, auc, HISTORY_FILE)

print ('Total time:', time.time() - start_time)
K.clear_session()
