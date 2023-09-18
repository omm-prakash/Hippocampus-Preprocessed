import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
plt.style.use("ggplot")

def plot_history(history, epoch, metric):
    train = history.history[metric]
    val = history.history['val_'+metric]

    plt.figure(figsize = (10,5))# (x,y)
    plt.plot(train,label = f'train {metric}')
    plt.plot(val,label = f'val {metric}')
#     plt.grid(True)
    plt.legend()
    plt.xticks(np.arange(1,epoch,step = 1))
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title(metric)

def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)


def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)


def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou

def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)
