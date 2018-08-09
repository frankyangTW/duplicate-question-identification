import tensorflow as tf
import keras.backend as K

#%% define metrics
def precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(tf.multiply(y_true, y_pred))
    fp = K.sum(tf.multiply(1 - y_true, y_pred))
    return (tp / (K.epsilon() + tp + fp))

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(tf.multiply(y_true, y_pred))
    fp = K.sum(tf.multiply(1 - y_true, y_pred))
    fn = K.sum(tf.multiply(1 - y_pred, y_true))
    precision = tp / (K.epsilon() + tp + fp)
    recall = tp / (K.epsilon() + tp + fn)
    f = 2 * precision * recall / (K.epsilon() + precision + recall)
    return f
def MCC(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(tf.multiply(y_true, y_pred))
    tn = K.sum(tf.multiply(1 - y_true, 1 - y_pred))
    fp = K.sum(tf.multiply(1 - y_true, y_pred))
    fn = K.sum(tf.multiply(1 - y_pred, y_true))
    n = tp + fp + tn + fn
    s = (tp + fn) / n
    p = (tp + fp) / n
    mcc =  (tp / n - s * p)/ (K.epsilon() + K.sqrt(s*p*(1-s)*(1-p)))
    return mcc
def FPR(y_true, y_pred):
    y_pred = K.round(y_pred)
    fp = K.sum(tf.multiply(1 - y_true, y_pred))
    n = tf.shape(y_pred)[0]
    return fp / tf.cast(n, tf.float32)
def TPR(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(tf.multiply(y_true, y_pred))
    n = tf.shape(y_pred)[0]
    return tp / tf.cast(n, tf.float32)
def FNR(y_true, y_pred):
    y_pred = K.round(y_pred)
    fn = K.sum(tf.multiply(1 - y_pred, y_true))
    n = tf.shape(y_pred)[0]
    return fn / tf.cast(n, tf.float32)
def TNR(y_true, y_pred):
    y_pred = K.round(y_pred)
    tn = K.sum(tf.multiply(1 - y_true, 1 - y_pred))
    n = tf.shape(y_pred)[0]
    return tn / tf.cast(n, tf.float32)
















