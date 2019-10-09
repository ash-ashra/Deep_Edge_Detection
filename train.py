from bsds500 import BSDS500
import seaborn
from sklearn.model_selection import train_test_split
import imgaug as ia
import imgaug.augmenters as iaa

from skimage.transform import resize
from skimage.io import imsave
from scipy.misc import imread
from scipy.io import loadmat
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import backend as K
from imutils.paths import list_images
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


warnings.filterwarnings('ignore')

K.set_image_data_format('channels_last')

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet(img_rows, img_cols, channels):
    inputs = Input((img_rows, img_cols, channels))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=5e-5),
                  # loss='binary_crossentropy',
                  loss=dice_coef_loss,
                  # metrics=['binary_crossentropy'])
                  metrics=[dice_coef])

    return model


TARGET_SHAPE = (256, 256)

# bsds = BSDS500(target_size=TARGET_SHAPE)
# ids, contours_train, sgmnts, train_images = bsds.get_train()
# ids, contours_test, sgmnts, test_images = bsds.get_test()
# ids, contours_val, sgmnts, val_images = bsds.get_val()
#
# C = np.concatenate([contours_val, contours_test, contours_train])
# np.save('./../C.npy', C)
C = np.load('./../C.npy')
print(C.shape)
print(np.max(C))

# I = np.concatenate([val_images, test_images, train_images])
# np.save('./../I.npy', I)
I = np.load('./../I.npy')

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


I = rgb2gray(I)
I = I[..., np.newaxis]


print(I.shape)
print(np.max(I))
# Split the dataset
random_seed = 21
X_train, X_test, Y_train, Y_test = train_test_split(
    I, C, test_size=0.2, random_state=random_seed)



# Define a function to augment input and labels :
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.


def sometimes(aug): return iaa.Sometimes(0.5, aug)


# Define our sequence of augmentation steps that will be applied to every image
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate by -20 to +20 percent (per axis)
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            # use nearest neighbour or bilinear interpolation (fast)
            order=[0, 1],
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            mode=ia.ALL
        ))
    ],
    random_order=True
)


def augment(batch_x, batch_y):

    # augmentation using imgaug library
    batch_x *= 255
    batch_y *= 255
    batch_x = batch_x.astype(np.uint8)
    batch_y = batch_y.astype(np.int32)

    batch_x, batch_y = seq(images=batch_x, segmentation_maps=batch_y)

    # change type
    batch_x = batch_x.astype(np.float64)/255
    batch_y = batch_y.astype(np.float64)/255

    return batch_x, batch_y

# Define the denerator that prepares the batch


def data_generator(X, Y, batch_size=64):
    while True:
        # Select files indices for the batch
        indxs = np.random.choice(a=range(len(X)),
                                 size=batch_size)
        batch_x = X[indxs]
        batch_y = Y[indxs]

        batch_x, batch_y = augment(batch_x, batch_y)

        yield batch_x, batch_y


unet = get_unet(TARGET_SHAPE[0], TARGET_SHAPE[1], 1)
model_checkpoint = ModelCheckpoint('./../best_weights3.h5',
                                   monitor='val_dice_coef_loss',
                                   save_best_only=True)
csv_callback = CSVLogger('history3.log', append=True)

BS = 64
my_gen = data_generator(X_train, Y_train, batch_size=BS)
steps_per_epoch = len(X_train) // BS




H = unet.fit_generator(my_gen, steps_per_epoch=steps_per_epoch,
                   epochs=200, verbose=1,
                   callbacks=[csv_callback, model_checkpoint])


unet.save('./../last_unet3.h5')

f = pd.DataFrame(H.history)[['dice_coef_loss', 'val_dice_coef_loss']].plot(figsize=(12, 9))
plt.title('model training 3')
plt.ylabel('binary_crossentropy_loss')
plt.xlabel('epochs')
plt.savefig('result3.png')
