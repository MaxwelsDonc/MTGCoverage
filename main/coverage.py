"""
define coverage
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import copy
import para
import heapq

par = para.Para()


def cal_neulev_cov(layers_output, mfr):
    """
    计算神经元覆盖率
    :type neurons_output:
    :param layers_output:
    :param mfr:
    :return:
    """
    # 把层输出转换成神经元输出
    neurons_output = None
    for i in range(len(layers_output)):
        temp = np.transpose(layers_output[i])
        # 合并神经元，位置上确定的。
        if neurons_output is None:
            neurons_output = temp
        else:
            neurons_output = np.concatenate((neurons_output, temp))
    # 开始计算
    MultiSecNeuCovNum, UpConNeuNum, LowConNeuNum = 0, 0, 0
    for n in range(len(neurons_output)):
        n_msnc, n_ucn, n_lcn = 0, 0, 0
        # generate k-multisection zoom
        n_max, n_min = mfr[n][0], mfr[n][1]
        # n_zoom = np.linspace(n_min, n_max, par.k_section + 1)
        n_gap = (n_max - n_min) / par.k_section
        n_index = []
        # get the output of neuron n
        neuron_output = neurons_output[n]
        neuron_output = list(set(neuron_output))
        # judge the value is belong which sub-zoom
        for value in neuron_output:
            if value < n_min:
                n_lcn = 1
            elif value > n_max:
                n_ucn = 1
            else:
                if n_gap == 0:
                    continue
                gap_index = (value - n_min) // n_gap
                n_index.append(gap_index)
        # 计算有多少小节被覆盖到了
        n_msnc = len(set(n_index))
        MultiSecNeuCovNum += n_msnc
        UpConNeuNum = UpConNeuNum + n_ucn
        LowConNeuNum = LowConNeuNum + n_lcn
    MultiSecNeuCov = MultiSecNeuCovNum / (par.k_section * len(neurons_output))
    NeuBoundCov = (UpConNeuNum + LowConNeuNum) / (2 * len(neurons_output))
    StrNeuActCov = UpConNeuNum / len(neurons_output)

    return MultiSecNeuCov, NeuBoundCov, StrNeuActCov


def cal_laylev_cov(layers_output, mfr):
    NeuNum = len(mfr)
    TopKNeuNum = 0
    # 进行相关数据的计算，每一层每个样本激活值最高的几个神经元index都会被存储
    # 每一层
    top_k_neuron_layers = []
    for layer_output in layers_output:
        top_k_neuron_layer = []
        # 每一个样本
        for sample_output in layer_output:
            top_k_neuron_layer_sample = list(map(list(sample_output).index, heapq.nlargest(par.top_k, sample_output)))
            top_k_neuron_layer.append(top_k_neuron_layer_sample)

        top_k_neuron_layers.append(top_k_neuron_layer)
    # 计算TKNCov
    for top_k_neuron_layer in top_k_neuron_layers:
        temp = []
        for top_k_neuron_layer_sample in top_k_neuron_layer:
            temp.extend(top_k_neuron_layer_sample)
        TopKNeuNum += len(set(temp))

    # 计算TKNPat
    TNKPatList = []
    # 每个样本
    for sample_index in range(len(layers_output[0])):
        sample_topk_pattern = []
        for layer_index in range(len(layers_output)):
            temp = top_k_neuron_layers[layer_index][sample_index]
            sample_topk_pattern.extend(temp)
        TNKPatList.append(hash(str(sample_topk_pattern)))

    TKNCov = TopKNeuNum / NeuNum
    TNKPat = len(set(TNKPatList))
    return TKNCov, TNKPat
