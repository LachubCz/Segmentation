import os
import cv2
import numpy as np

from os import listdir
from sklearn.model_selection import train_test_split


def get_img(data_path):
    img = cv2.imread(os.path.join(os.getcwd(), data_path), flags=cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f'error unable to read {os.path.join(os.getcwd(), data_path)}')

    return img


def get_dataset(dataset_path):
    images = listdir(dataset_path)
    images = [i for i in images if '_mask' not in i]
    X = []
    Y = []
    for img in images:
        img_path = os.path.join(dataset_path, img)

        x_img = get_img(img_path)
        y_img = get_img(img_path.replace('.jpg', '_mask.jpg'))

        X.append(x_img)
        Y.append(y_img)
    X = np.array(X)
    Y = np.array(Y).astype(np.float32)

    return X, Y  # X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
