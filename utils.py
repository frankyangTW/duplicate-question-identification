from keras.callbacks import Callback
import keras.backend as K
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import roc_auc_score

class AucCallback(Callback):

    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.auc = []
        self.val_auc = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print ('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)),
                        str(round(roc_val, 4))))
        self.auc.append(round(roc, 4))
        self.val_auc.append(round(roc_val, 4))
        return


def limitGPU():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))
    return


def loadDataset(augmentation=False, small_set=False):
    train_x1 = np.load('datasets/train_x1.npy')
    train_x2 = np.load('datasets/train_x2.npy')
    train_features = np.load('datasets/train_HCF.npy')
    train_y = np.load('datasets/train_y.npy')
    if augmentation:
        train_x1_aug = np.load('datasets/train_x1_augmented.npy')
        train_x2_aug = np.load('datasets/train_x2_augmented.npy')
        train_features_aug = np.load('datasets/train_HCF_augmented.npy')
        train_y_aug = np.load('datasets/train_y_augmented.npy')
        train_x1 = np.vstack([train_x1, train_x1_aug])
        train_x2 = np.vstack([train_x2, train_x2_aug])
        train_features = np.vstack([train_features, train_features_aug])
        train_y = np.append(train_y, train_y_aug)
    if small_set:
        train_x1 = train_x1[:30000]
        train_x2 = train_x2[:30000]
        train_features = train_features[:30000]
        train_y = train_y[:30000]
    valid_x1 = np.load('datasets/valid_x1.npy')
    valid_x2 = np.load('datasets/valid_x2.npy')
    valid_y = np.load('datasets/valid_y.npy')
    valid_features = np.load('datasets/valid_HCF.npy')
    return train_x1, train_x2, train_features, train_y, valid_x1, valid_x2, valid_y, valid_features

def load_embedding_matrix():
    matrix_file = 'datasets/embedding_matrix.npy'
    embedding_matrix = np.load(matrix_file)
    vocab_size = np.shape(embedding_matrix)[0]
    return (embedding_matrix, vocab_size)

def saveHistory(history, auc, HISTORY_FILE, *argv):
    history.history['auc'] = auc.auc
    history.history['val_auc'] = auc.val_auc
    if len(argv) > 0:
        history_2 = argv[0]
        auc_2 = argv[1]
        history_2.history['auc'] = auc_2.auc
        history_2.history['val_auc'] = auc_2.val_auc
        for key in history.history.keys():
            history.history[key] = np.append(history.history[key], history_2.history[key])

    my_file = Path(HISTORY_FILE)
    if my_file.is_file():
        with open(HISTORY_FILE, 'rb') as f:
            old_history = pickle.load(f)
            for key in old_history.keys():
                history.history[key] = np.append(old_history[key],
                        history.history[key])
    with open(HISTORY_FILE, 'wb+') as f:
        pickle.dump(history.history, f)
        print ('History Saved')
    return

def layer_norm(x):
    return tf.contrib.layers.layer_norm(x)


