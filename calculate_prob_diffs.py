import tensorflow as tf
import numpy as np
import os
import pickle
import copy
import skimage.io
import skimage.segmentation
from skimage import filters
import pandas as pd
import cv2
import random
import sys

from numpy import asarray
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

model = keras.applications.inception_v3.InceptionV3(weights = 'imagenet')
img_size = (299, 299)
preprocess_input = tf.keras.applications.inception_v3.preprocess_input
decode_predictions = tf.keras.applications.inception_v3.decode_predictions

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def select_n_random_pixels(img, n):
    width, height,_ = img.shape
    rows, columns, _ = img.shape

    pixel_indices = [(x, y) for x in range(width) for y in range(height)]
    selected_pixels = random.sample(pixel_indices, n)
    return selected_pixels


master_dir_path = "cifar10_test/"
img_dirs = os.listdir(master_dir_path)
num_pxls = 50

for img_dir in img_dirs:
    file_name_probs = "inception_cifar10_" + img_dir + "_prob.pkl"
    file_probs = open(file_name_probs, "wb")

    probs_list = []
    img_paths = os.listdir(master_dir_path + img_dir)
    for img_path in img_paths:
        img_full_path = master_dir_path + img_dir + "/" + img_path
        img = tf.keras.preprocessing.image.load_img(img_full_path, target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = tf.keras.applications.xception.preprocess_input(img)
            pred0 = model.predict(np.array([img]), verbose=0)
            top_pred_class = pred0[0].argsort()[-1:][::-1]
            prob0 = pred0[0][top_pred_class]

            pert_pxl_list = select_n_random_pixels(img, num_pxls)
            probs_arr = []
            sigmas = [0.2, 0.6, 1]
            for sigma in sigmas:
                probs = []
                blur_img = filters.gaussian(img, sigma, multichannel=True)
                perturbed_images = []
                for pxl in pert_pxl_list:
                    mask = np.ones(img_size)
                    mask[pxl] = 0
                    mask3d = cv2.merge((mask, mask, mask))
                    perturbed_image = np.where(mask3d == np.array([0.0, 0.0, 0.0]), blur_img, img)
                    perturbed_images.append(perturbed_image)

                preds = model.predict(np.array(perturbed_images), verbose=False)
                probs_arr = a = preds[:, top_pred_class]
                # probs_arr = np.array(probs)
                probs_arr = np.append(probs_arr, sigma)
                probs_arr = np.append(probs_arr, prob0)
                probs_arr = np.append(probs_arr, img_path)
                # (decode_predictions(preds, top=1)[0])[0][1]
                # print(probs_arr)
                probs_list.append(probs_arr)

    pickle.dump(probs_list, file_probs)
    file_probs.close()
