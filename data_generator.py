import os
import numpy as np
import nibabel as nib
import keras
import random

class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 image_names,
                 data_path,
                 data_name='Dryad',
                 batch_size=2,
                 dim=(58,64),
                 bands=64,
                 n_classes=2,
                 shuffle=True,
                 augment=False,
                 augmentation_prob=0.5
                ):        
        self.image_names = image_names
        self.data_path = data_path
        self.data_name = data_name
        self.batch_size = batch_size
        self.dim = dim
        self.bands = bands
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment=augment,
        self.augmentation_prob=augmentation_prob
        self.on_epoch_end()
        
    # Denotes the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.image_names) / self.batch_size))

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # Generate one batch of data
    # Generate indexes of the batch
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        batch_images = [self.image_names[k][1] for k in indexes]
        # Generate data
        X, y = self.__data_generation(batch_images)
        return X, y

    # Generates data containing batch_size samples 
    # X : (n_samples, *patch, bands)       
    def __data_generation(self, batch_images):
        X = np.empty((self.batch_size, *self.dim, self.bands))
        y = np.empty((self.batch_size, *self.dim, self.bands))
        for i,image_path in enumerate(batch_images):
            # Load image
            imageObj = nib.load(os.path.join(self.data_path,self.data_name,image_path.file))
            image = imageObj.get_fdata()
            
            # Load annotation data
            annotationObj = nib.load(os.path.join(self.data_path,self.data_name,image_path['gt']))
            annotation = annotationObj.get_fdata()
            
            # Apply data augmentation if augmentation is enabled
            if self.augment:
                if random.random() < self.augmentation_prob:
                    image = np.fliplr(image)  # Horizontal flip with augmentation_prob probability
                    annotation = np.fliplr(annotation)
                else:
                    image = np.flipud(image)  # Vertical flip with augmentation_prob probability
                    annotation = np.flipud(annotation)
            X[i] = image
            y[i] = annotation
            
        X = X.reshape(-1, *self.dim, self.bands, 1).copy()
        y = y.reshape(-1, *self.dim, self.bands, 1).copy()
        return X,y 