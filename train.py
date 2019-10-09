import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imutils.paths import list_images
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from scipy.io import loadmat
from scipy.misc import imread
from skimage.io import imsave
from skimage.transform import resize

warnings.filterwarnings('ignore')

K.set_image_data_format('channels_last')



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
                      loss='binary_crossentropy',
                      metrics=['binary_crossentropy'])


    return model

from bsds500 import BSDS500
TARGET_SHAPE = (256, 256)

bsds = BSDS500(target_size=TARGET_SHAPE)
ids, contours_train, sgmnts, train_images = bsds.get_train()
ids, contours_test, sgmnts, test_images = bsds.get_test()
ids, contours_val, sgmnts, val_images = bsds.get_val()

C = np.concatenate([contours_val, contours_test, contours_train])
I = np.concatenate([val_images, test_images, train_images])


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

I = rgb2gray(I)
I = I[...,np.newaxis]


model_checkpoint = ModelCheckpoint('./../best_weights.h5',
                                   monitor='val_loss',
                                   save_best_only=True)

csv_callback = CSVLogger('history.log', append=True)
unet = get_unet(TARGET_SHAPE[0], TARGET_SHAPE[1], 1)
H = unet.fit(I, C, verbose=1, epochs=100, validation_split=0.1, callbacks=[csv_callback, model_checkpoint])

from keras.models import load_model
unet.save('./../last_unet.h5')

import seaborn
f = pd.DataFrame(H.history)[['loss','val_loss']].plot(figsize=(12,9))
plt.title('model training')
plt.ylabel('binary_crossentropy_loss')
plt.xlabel('epochs')
plt.savefig('result1.png')
