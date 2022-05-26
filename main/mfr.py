"""
this script is used to get mftr
neuron in CNN is defined as a channel.
"""
import os
import threading
import time

import numpy as np
from tensorflow import keras
from tqdm import tqdm


def start_thread(thread_list):
    """
    :param thread_list: the list of thread
    """
    for t in thread_list:
        t.start()


def join_thread(thread_list):
    """
    :param thread_list: the list of thread
    """
    for t in thread_list:
        t.join()


def merge_mfr(path, save_name=None):
    """

    :param path: the dir path of mfr_section
    :param save_name: save name
    """
    mfr = None
    name_list = os.listdir(path)
    for name in name_list:
        load_numpy = np.load(path + '/' + name)
        half = len(load_numpy) // 2
        load_numpy = np.concatenate((load_numpy[0:half], load_numpy[half:2 * half]), axis=1)
        if mfr is None:
            mfr = load_numpy
            continue
        else:
            temp = np.concatenate((mfr, load_numpy), axis=1)
            mfr_max = temp.max(axis=1)
            mfr_min = temp.min(axis=1)
    mfr_max = np.expand_dims(mfr_max, axis=1)
    mfr_min = np.expand_dims(mfr_min, axis=1)
    mfr = np.concatenate((mfr_max, mfr_min), axis=1)
    return mfr


def getmfr(model_name, data_name=None, data_dir=None):
    """
    :param model_name: the name of the model
    :param data_name: the name of the datasete (MNIST, ImageNet)
    :return: return the mfr
    """
    if model_name in ['LeNet1', 'LeNet4', 'LeNet5']:
        # if the model is constructed by ourselves
        model_path = f'../data/{model_name}/{model_name}{data_name}.h5'
        save_path = f'../data/{model_name}/{model_name}{data_name}.npy'
        model = keras.models.load_model(model_path)
        if data_name == 'MNIST':
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            # 归一化
            x_train = x_train.astype(np.float)
            x_test = x_test.astype(np.float)
            x_train /= 255
            x_test /= 255
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
        elif data_name == 'fashionmnist':
            (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
            # 归一化
            x_train = x_train.astype(np.float)
            x_test = x_test.astype(np.float)
            x_train /= 255
            x_test /= 255
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
        # get the layer outputs
        model_input = model.layers[0].input
        model_output = [
            layer.output for layer in model.layers if 'flatten' not in layer.name]
        get_layers_output = keras.backend.function([model_input], model_output)
        layers_output = get_layers_output(x_train)
        # get the max and min values of the neurons
        layers_mfr = None
        for layer_output in layers_output:
            if len(layer_output.shape) > 2:
                axis = tuple((i for i in range(1, len(layer_output.shape) - 1)))
                layer_output = layer_output.mean(axis=axis)
            layer_max = layer_output.max(axis=0)
            layer_min = layer_output.min(axis=0)
            layer_max = np.expand_dims(layer_max, axis=1)
            layer_min = np.expand_dims(layer_min, axis=1)
            layer_mfr = np.concatenate((layer_max, layer_min), axis=1)
            if layers_mfr is None:
                # uninitialized
                layers_mfr = layer_mfr
            else:
                layers_mfr = np.concatenate((layers_mfr, layer_mfr))
        np.save(save_path, layers_mfr)
        return None
    elif model_name in ['ResNet50', 'VGG19']:
        # get the names of the images
        data_name_list = os.listdir(data_dir)
        # load the model
        if model_name == 'ResNet50':
            model = keras.applications.ResNet50()
        elif model_name == 'VGG19':
            model = keras.applications.VGG19()
        # construct the layer output get-function
        model_input = model.layers[0].input
        model_output = [layer.output for layer in model.layers if
                        ('flatten' not in layer.name and 'input' not in layer.name)]
        get_layers_output = keras.backend.function([model_input], model_output)
        # begain to load the data
        batch = 10
        mfr_max, mfr_min = None, None
        for data_name in tqdm(data_name_list):
            data_path = data_dir + '/' + data_name
            x_train = np.load(data_path)
            x_train = keras.applications.vgg19.preprocess_input(x_train)
            beg, end, length = 0, batch, len(x_train)
            while beg < length:
                layers_max, layers_min = None, None
                x_batch = x_train[beg:min(end, length)]
                print(f'\r############{end}/{length}############', end='')
                layers_output_batch = get_layers_output(x_batch)
                # get the mfr_batch
                for layer_output in layers_output_batch:
                    if len(layer_output.shape) > 2:
                        axis = tuple((i for i in range(1, len(layer_output.shape) - 1)))
                        layer_output = layer_output.mean(axis=axis)
                    layer_max = layer_output.max(axis=0)
                    layer_min = layer_output.min(axis=0)
                    layer_max = np.expand_dims(layer_max, axis=1)
                    layer_min = np.expand_dims(layer_min, axis=1)
                    # initialized
                    if layers_max is None and layers_min is None:
                        layers_max = layer_max
                        layers_min = layer_min
                    else:
                        layers_max = np.concatenate((layers_max, layer_max))
                        layers_min = np.concatenate((layers_min, layer_min))
                # we have get one batch mfr_max and mfr_min(layers_max,layers_min)
                if mfr_min is None and mfr_max is None:
                    mfr_max = layers_max
                    mfr_min = layers_min
                else:
                    mfr_max = np.expand_dims(np.concatenate(
                        (layers_max, mfr_max), axis=1).max(axis=1), axis=1)
                    mfr_min = np.expand_dims(np.concatenate(
                        (layers_min, mfr_min), axis=1).min(axis=1), axis=1)
                beg += batch
                end += batch
        mfr = np.concatenate((mfr_max, mfr_min), axis=1)
        np.save(
            f'../data/{model_name}/{model_name}{data_name}{data_dir[-1]}.npy', mfr)


if __name__ == '__main__':
    # mfr = merge_mfr('../data/VGG19/')
    # np.save('../data/VGG19/VGG19ImageNet.npy', mfr)
    # mfr = merge_mfr('../data/ResNet50/')
    # np.save('../data/ResNet50/ResNet50ImageNet.npy', mfr)
    getmfr('LeNet1', 'fashionmnist')
