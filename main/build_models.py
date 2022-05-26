import os.path

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset_name_set = 'fashion_mnist'


def lenet1(x_train, y_train, x_test, y_test, dataset_name):
    """
    搭建函数lenet1的模型
    Returns:
    """
    # build

    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(5, 5), padding="valid", activation="tanh", input_shape=[32, 32, 3]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=12, kernel_size=(5, 5), padding="valid", activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(Dense(120, activation="tanh"))
    # model.add(Dense(84, activation="tanh"))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
    # train
    model.fit(x_train, y_train, batch_size=32, epochs=30)
    # save
    model.evaluate(x_test, y_test)
    save_or_not = input("是否进行保存[y/n]:\n")
    if save_or_not == 'y':
        model.save(f"../data/LeNet1/LeNet1_{dataset_name}.h5")
    else:
        print("模型没有训练满意")
        return 0
    return model


def load_data(dataset_name_para):
    """
    载入相关的数据集数据
    Args:
        dataset_name_para:数据集合的名字

    Returns:返回训练集和测试集合

    """
    x_train_arr, y_train_arr, x_test_arr, y_test_arr = None, None, None, None
    if dataset_name_para == 'fashionmnist':
        (x_train_arr, y_train_arr), (x_test_arr, y_test_arr) = tf.keras.datasets.fashion_mnist.load_data()
        assert x_train_arr.shape == (60000, 28, 28)
        assert x_test_arr.shape == (10000, 28, 28)
        assert y_train_arr.shape == (60000,)
        assert y_test_arr.shape == (10000,)
        # normailize
        x_train_arr = x_train_arr / 255
        x_test_arr = x_test_arr / 255
        # 1->[0,1,0,0,0,0,0,0,0,0]
        y_train_arr = tf.keras.utils.to_categorical(y_train_arr, num_classes=None, dtype='float32')
        y_test_arr = tf.keras.utils.to_categorical(y_test_arr, num_classes=None, dtype='float32')
    elif dataset_name_para == 'cifar10':
        # if not os.path.exists("../data/cifar10"):
        #     os.mkdir("../data/cifar10")
        (x_train_arr, y_train_arr), (x_test_arr, y_test_arr) = tf.keras.datasets.cifar10.load_data()
            # 转化为灰度图像
            # x_train_grey = []
            # for image in tqdm(x_train_arr):
            #     image_grey = tf.image.rgb_to_grayscale(image)
            #     x_train_grey.append(image_grey)
            #
            # x_test_grey = []
            # for image in tqdm(x_test_arr):
            #     image_grey = tf.image.rgb_to_grayscale(image)
            #     x_test_grey.append(image_grey)
            # # 替换成训练集合
            # x_train_grey = np.array(x_train_grey)
            # x_test_grey = np.array(x_test_grey)
            #
        x_train_arr = x_train_arr / 255
        x_test_arr = x_test_arr / 255
        # 1->[0,1,0,0,0,0,0,0,0,0]
        y_train_arr = tf.keras.utils.to_categorical(y_train_arr, num_classes=None, dtype='float32')
        y_test_arr = tf.keras.utils.to_categorical(y_test_arr, num_classes=None, dtype='float32')
            #
            # np.save( "../data/cifar10/x_train_gray.npy",x_train_grey)
            # np.save( "../data/cifar10/x_test_gray.npy",x_test_grey)
            # np.save( "../data/cifar10/y_train_gray.npy",y_train_grey)
            # np.save("../data/cifar10/y_test_gray.npy",y_test_grey)
        #
        # x_train_arr = np.load("../data/cifar10/x_train_gray.npy")
        # x_test_arr = np.load("../data/cifar10/x_test_gray.npy")
        # y_train_arr = np.load("../data/cifar10/y_train_gray.npy")
        # y_test_arr = np.load("../data/cifar10/y_test_gray.npy")

    return x_train_arr, y_train_arr, x_test_arr, y_test_arr


if __name__ == '__main__':
    dataset_name = 'cifar10'
    x_train, y_train, x_test, y_test = load_data(dataset_name)
    model = lenet1(x_train, y_train, x_test, y_test, dataset_name)
