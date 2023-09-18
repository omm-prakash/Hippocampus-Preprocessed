from keras import layers
from keras.models import Model

# Defining 3D-Unet architecture
kernel_initializer = 'he_uniform' #Try others if you want
def down_block(input, kernels, drop_ratio=0):
    conv = layers.Conv3D(kernels, (3, 3, 2), activation='relu', kernel_initializer=kernel_initializer, padding='same')(input)
    d = layers.Dropout(drop_ratio)(conv)
    conv = layers.Conv3D(kernels, (3, 3, 2), activation='relu', kernel_initializer=kernel_initializer, padding='same')(d)
    p = layers.MaxPooling3D((2, 2, 2))(conv)
    #print(input.shape, p.shape)
    return (p,conv)

def up_block(input, kernels, drop_ratio, concat_layer):
    conv = layers.Conv3DTranspose(kernels, (2, 2, 2), strides=(2, 2, 2), padding='same')(input)
    concat = layers.concatenate([conv, concat_layer])
    conv = layers.Conv3D(kernels, (3, 3, 2), activation='relu', kernel_initializer=kernel_initializer, padding='same')(concat)
    d = layers.Dropout(drop_ratio)(conv)
    conv = layers.Conv3D(kernels, (3, 3, 2), activation='relu', kernel_initializer=kernel_initializer, padding='same')(d)
    #print(input.shape, conv.shape)
    return conv
  
def unet3D(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS):
    inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs
    # Contraction path
    (p1,c1) = down_block(s, 16, 0.1)
    (p2,c2) = down_block(p1, 32, 0.1)
    (p3,c3) = down_block(p2, 64, 0.2)
    #(p4,c4) = down_block(p3, 128, 0.2)
    
    c5 = layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    # Expansive path 
    # c6 = up_block(c5, 128, 0.2, c4)
    c7 = up_block(c5, 64, 0.2, c3)
    c8 = up_block(c7, 32, 0.1, c2)
    c9 = up_block(c8, 16, 0.1, c1)

    output = layers.Conv3D(1, (1,1,1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[output], name='3D-UNET')
    
    # Compile model outside of this function to make it flexible. 
    return model