import warnings
import numpy as np
from scipy.misc import imread, imsave
from skimage import data
from skimage.transform import resize
from keras.models import load_model
import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# import matplotlib.pyplot as plt
import sys

warnings.filterwarnings('ignore')
coins = data.coins()


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

    target_size = model.input.__dict__['_keras_shape'][1:-1]

    im_resize = resize(im, target_size)
    im = np.expand_dims(im_resize, 0)
    preds = model.predict(im)
    pred = preds[:, :, :, 0][0]


    # canny_pred = blurr_canny(float_image_to_uint8(im_resize))

    return pred

if __name__ == '__main__':
    file_name = sys.argv[1]
    unet = load_model('./../best_weights.h5')
    c = predict_custom_image(file_name, unet)

    plt.figure(figsize=(12,12))
    plt.imshow(c)
    plt.axis('off')
    plt.savefig('edges.jpg')


#docker run -v /path/to/file1:/path/to/file.txt -t boot:latest python boot.py file1.txt
