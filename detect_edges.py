import warnings
import numpy as np
from scipy.misc import imread, imsave
from skimage import data
from skimage.transform import resize
from keras.models import load_model
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import backend as K

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# import matplotlib.pyplot as plt
import sys

warnings.filterwarnings('ignore')
coins = data.coins()

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

# def to_rgb1(im):
#     # I think this will be slow
#     w, h = im.shape
#     ret = np.empty((w, h, 3), dtype=np.uint8)
#     ret[:, :, 0] = im
#     ret[:, :, 1] = im
#     ret[:, :, 2] = im
#     return ret
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# def blurr_canny(im, sigma=0.2):
#     blur = cv2.GaussianBlur(im, (5, 5), 0)
#     return auto_canny(blur)


def float_image_to_uint8(im):
    return (im * 255).round().astype('uint8')


def predict_custom_image(image=None, model=None):
    if isinstance(image, str):
        im = imread(image)
    else:
        im = image

    if len(im.shape) == 3:
        im = rgb2gray(im)
        im = im[...,np.newaxis]

    target_size = (256, 256)

    im_resize = resize(im, target_size)
    im = np.expand_dims(im_resize, 0)
    preds = model.predict(im)
    pred = preds[:, :, :, 0][0]


    # canny_pred = blurr_canny(float_image_to_uint8(im_resize))

    return pred

if __name__ == '__main__':
    file_name = sys.argv[1]
    TARGET_SHAPE = (256, 256)
    unet = get_unet(TARGET_SHAPE[0], TARGET_SHAPE[1], 1)
    unet.load_weights('./../last_unet3.h5')
    c = predict_custom_image(file_name, unet)

    plt.figure(figsize=(12,12))
    plt.imshow(c)
    plt.axis('off')
    plt.savefig('edges.jpg')


#docker run -v /path/to/file1:/path/to/file.txt -t boot:latest python boot.py file1.txt
