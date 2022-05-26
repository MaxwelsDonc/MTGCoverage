"""
this script is used to store the mr of this experiment
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import copy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transform(image, alpha, sigma, random_state=None):
    """
    :param image:
    :param alpha:
    :param sigma:
    :param random_state:
    :return:
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    shape = image.shape
    shape_size = shape[:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def grey_transform(imageorg, k, b):
    """
    :param imageorg:
    :param k:
    :param b:
    :return:
    """
    image = copy.deepcopy(imageorg)
    if image.shape[-1] == 1:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j][0] = image[i][j][0] * k + b
    elif image.shape[-1] == 3:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for p in range(image.shape[2]):
                    # 图片未归一化
                    image[i][j][p] = image[i][j][p] * k + (1 - k) * 255

    return image


def pepper_noise(imageorg):
    """

    :param imageorg:
    :return:
    """
    image = copy.deepcopy(imageorg)
    rate = 10 / (28 * 28)
    # rate = 0
    size = image.shape[0] * image.shape[1]
    n_pepper = int(rate * size)
    for __ in range(n_pepper):
        randx = np.random.randint(1, image.shape[0] - 1)  # 生成一个 1 至 img_r-1 之间的随机整数
        randy = np.random.randint(1, image.shape[1] - 1)  # 生成一个 1 至 img_c-1 之间的随机整数
        if image.shape[-1] == 1:
            if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
                image[randx, randy] = 0
            else:
                image[randx, randy] = 1
        elif image.shape[-1] == 3:
            if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
                image[randx, randy] = 0
            else:
                image[randx, randy] = 255
    return image
