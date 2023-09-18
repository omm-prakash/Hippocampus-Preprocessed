import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import nibabel as nib

data_path = os.getcwd()
data_name = 'Dryad'
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

def find_gt(name):
    return name[:5]+'_gt'+name[5:]

# Define a function to visualize the prediction data
def pred_3dimage(Image, Layer, predictions, test):
    imageObj = nib.load(os.path.join(data_path,data_name,Image))
    gt_imageObj = nib.load(os.path.join(data_path,data_name,find_gt(Image)))
    image = imageObj.get_fdata()
    imageGT = gt_imageObj.get_fdata()
    pred = predictions[test[test.file==Image].index[0]]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image[:, :, Layer], cmap='plasma');
    plt.title(f'Explore Layers of {data_name}')
    plt.xlabel(Image)
    
    plt.subplot(132)
    plt.imshow(imageGT[:, :, Layer], cmap='plasma');
    plt.title('Ground Truth')
    plt.xlabel(find_gt(Image))
    
    plt.subplot(133)
    plt.imshow(pred[:, :, Layer], cmap='plasma');
    plt.title('Model Prediction')
    return
