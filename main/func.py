"""
这个文件是相关数据的载入文件
"""
import os

import tensorflow as tf
import numpy as np
import copy
import mr as mrp
import coverage as cov
import para
import openpyxl as opxl

par = para.Para()


def load_mgs(model_name, data_name=None, mr=False):
    if mr is False:
        # the path
        mgs_satisfied_path = f'../data/{model_name}/MGs_satisfied{data_name}.npy'
        mgs_violation_path = f'../data/{model_name}/MGs_violation{data_name}.npy'
        # load mgs
        mgs_satisfied = np.load(mgs_satisfied_path, allow_pickle=True)
        mgs_violation = np.load(mgs_violation_path, allow_pickle=True)
    else:
        # the path
        mgs_satisfied_path = f'../data/{model_name}/mgs_sv_mr/mgs_s_mr{mr}.npy'
        mgs_violation_path = f'../data/{model_name}/mgs_sv_mr/mgs_v_mr{mr}.npy'
        # load mgs
        mgs_satisfied = np.load(mgs_satisfied_path, allow_pickle=True)
        mgs_violation = np.load(mgs_violation_path, allow_pickle=True)

    return mgs_satisfied, mgs_violation


def load_mgs_size(model_name, data_name=None):
    # the path
    mgs_satisfied_path = f'../data/{model_name}/mgs_satisfied_size.npy'
    mgs_violation_path = f'../data/{model_name}/mgs_violation_size.npy'
    # load mgs
    mgs_satisfied = np.load(mgs_satisfied_path, allow_pickle=True)
    mgs_violation = np.load(mgs_violation_path, allow_pickle=True)

    return mgs_satisfied, mgs_violation


def generate_xymgs(mgs_satisfied, mgs_violation, rate, size=1000):
    x_mgs, y_mgs = [], []
    mgs_satisfied = shuffle_data(mgs_satisfied)
    mgs_violation = shuffle_data(mgs_violation)
    mgs_violation_num = int(round(size // 2 * rate))
    mgs_satisfied_num = size // 2 - mgs_violation_num
    temp = mgs_violation[0:mgs_violation_num]
    temp = np.concatenate((temp, mgs_satisfied[0:mgs_satisfied_num]))
    for index in range(len(temp)):
        # # first time
        # if x_mgs is None:
        #     x_mgs = temp[index]['source'][0]
        #     y_mgs = temp[index]['source'][2]
        #     x_mgs = np.concatenate(x_mgs, temp[index]['follow'][0])
        #     y_mgs = np.concatenate(y_mgs, temp[index]['follow'][2])
        #     continue
        # # concatenate
        # x_mgs = np.concatenate(x_mgs, temp[index]['source'][0])  # x_source[index]
        # y_mgs = np.concatenate(y_mgs, temp[index]['source'][2])  # y_pre_source
        # x_mgs = np.concatenate(x_mgs, temp[index]['follow'][0])
        # y_mgs = np.concatenate(y_mgs, temp[index]['follow'][2])
        x_mgs.append(temp[index]['source'][0])
        y_mgs.append(temp[index]['source'][2][0])
        x_mgs.append(temp[index]['follow'][0])
        y_mgs.append(temp[index]['follow'][2][0])

    x_mgs = np.array(x_mgs)
    y_mgs = np.array(y_mgs)

    return x_mgs, y_mgs


def load_data(dataname):
    """
    :param dataname:数据集的名字
    :return:返回载入的数据
    """
    if dataname == "MNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # 归一化
        x_train = x_train / 255
        x_test = x_test / 255
        # 对数据维度进行相关的设置
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        # 为了方便后期处理对test数据进行排序
        index = np.array(range(0, len(y_test)))
        mappingpair = sorted(zip(y_test, index))
        mappingindex = [x[1] for x in mappingpair]
        mappingindex = np.array(mappingindex)
        x_test = x_test[mappingindex]
        y_test = y_test[mappingindex]
        # 对数据标签进行编码
        # y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        # y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    elif dataname == 'fashionmnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        # 归一化
        x_train = x_train / 255
        x_test = x_test / 255
        # 对数据维度进行相关的设置
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        # 为了方便后期处理对test数据进行排序
        index = np.array(range(0, len(y_test)))
        mappingpair = sorted(zip(y_test, index))
        mappingindex = [x[1] for x in mappingpair]
        mappingindex = np.array(mappingindex)
        x_test = x_test[mappingindex]
        y_test = y_test[mappingindex]
        # 对数据标签进行编码
        # y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        # y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    elif dataname == 'ImageNet':
        data_path = '../data/ImageNet/val_mat/'
        x_train, y_train = None, None
        x_test = np.load(data_path + 'val_mat.npy')
        y_test = np.load(data_path + 'label_mat.npy')

    return (x_train, y_train), (x_test, y_test)


def load_model(modelname):
    """
    :param modelname: 模型的名字
    :param dataname: 数据集的名字
    :return: 载入的模型
    """
    if modelname == 'VGG19':
        model = tf.keras.applications.VGG19()
        return model
    elif modelname == 'ResNet50':
        model = tf.keras.applications.ResNet50()
        return model
    else:
        path = f"../data/{modelname}/{modelname}MNIST.h5"
        if not os.path.exists(path):
            print("model path doesn't exsit")
            return None
        model = tf.keras.models.load_model(path)
        return model


def load_mfr(modelname):
    """
    :param modelname: 模型名字
    :param dataname: 数据名字
    :return: 返回载入的mfr
    """
    if modelname in ["VGG19", "ResNet50"]:
        dataname = "ImageNet"
    elif modelname in ["LeNet1", "LeNet4", "LeNet5"]:
        dataname = "MNIST"
    else:
        print("the modelname is not correct")
    path = f"../data/{modelname}/{modelname}{dataname}.npy"
    if not os.path.exists(path):
        print("mfr path doesn't exsit")
        return None
    mfr = np.load(path)
    return mfr


def rate_of_violation(y_label, x_data, model, rate):
    # # 1-model.evaluate()
    # if len(y_label[0]) > 1:
    #     y_pre = model.predict(x_data)
    #     y_label = np.argmax(y_label, axis=-1)
    #     y_pre = np.argmax(y_pre, axis=-1)
    #     falsenum = (y_label == y_pre).tolist().count(False)
    # else:
    # get the y_prediction

    beg, end = 0, 20
    y_pre = np.array([])
    # get the prediction results
    while end <= len(x_data):
        end = min(end, len(x_data))
        y_pre_sec_value = model.predict(x_data[beg:end])
        y_pre_sec = np.argmax(y_pre_sec_value, axis=-1)
        y_pre = np.concatenate((y_pre, y_pre_sec))
        beg = end
        end += 20

    # # cal the rate of violation, method-1
    # mr_num = int(len(y_label) * rate)
    # org_num = len(y_label) - mr_num
    # generate_quotient = round(mr_num / org_num)
    # generate_residue = round(mr_num % org_num)
    #
    # MTGs_violation_num = 0
    #
    # for k in range(generate_quotient):
    #     MTGs_violation_num_sec = (y_pre[0:org_num] == y_pre[k * org_num:(k + 1) * org_num]).tolist().count(False)
    #     MTGs_violation_num += MTGs_violation_num_sec
    # MTGs_violation_num_sec_res = (y_pre[0:generate_residue] == y_pre[
    #                                                            generate_quotient * org_num:generate_quotient * org_num + generate_residue]).tolist().count(
    #     False)
    # MTGs_violation_num += MTGs_violation_num_sec_res

    # cal the rate of violation, method-2
    size = len(y_label)

    mr_num = int(len(y_label) * rate)
    org_num = len(y_label) - mr_num

    # x_org = x_data[0:org_num]
    y_org_pre = y_pre[0:org_num]
    y_org_pre_check = y_org_pre.copy()
    times = size // org_num
    # 多了没关系
    for _ in range(times):
        y_org_pre_check = np.concatenate((y_org_pre_check, y_org_pre))

    MTGs_violation_num = (y_org_pre_check[0:size] == y_pre).tolist().count(False)

    rov = MTGs_violation_num / mr_num

    return rov


def shuffle_data(x_data, y_data=None, size=None):
    """
    the fuction is used to shuffle the data
    :param x_data: x data
    :param y_data: label data
    :param size: the size of return data
    :return: shuffled data
    """
    # 对数据进行打乱
    if size is None:
        size = len(x_data)
    index = np.arange(len(x_data))
    np.random.shuffle(index)
    x_data = x_data[index]
    if y_data is not None:
        y_data = y_data[index]
        return x_data[0:size], y_data[0:size]
    return x_data[0:size]


def mr_data_generate(x_org, y_org, mr):
    """
    this function is used to get the mr dataset
    :param x_org:
    :param y_org:
    :param mr:
    :return:
    """
    x_mr = x_org
    idg = tf.keras.preprocessing.image.ImageDataGenerator()
    # {0:Nochange, 1:Rotation, 2:shift, 3:zoom, 4:shear, 5:Elastic, 6:greychange, 7:peppernoise}
    for index in range(len(x_mr)):
        if mr == 0:  # nochange
            pass
        if mr == 1:  # 1:Rotation
            # theta = np.random.uniform(-30, 30)
            theta = 30
            x_mr[index] = idg.apply_transform(x_mr[index], {"theta": theta})
        elif mr == 2:  # 2:shift
            # tx = np.random.randint(-4, 4)
            # ty = np.random.randint(-4, 4)
            tx, ty = 4, -3
            x_mr[index] = idg.apply_transform(x_mr[index], {"tx": tx, "ty": ty})
        elif mr == 3:  # 3:zoom
            # flag = np.random.uniform(0, 2)
            # if flag >= 1:
            #     zx = np.random.uniform(1, 1.5)
            # else:
            #     zx = np.random.uniform(0.5, 1)
            # flag = np.random.uniform(0, 2)
            # if flag >= 1:
            #     zy = np.random.uniform(1, 1.5)
            # else:
            #     zy = np.random.uniform(0.5, 1)
            zx, zy = 1.5, 1.6
            x_mr[index] = idg.apply_transform(x_mr[index], {"zx": zx, "zy": zy})
        elif mr == 4:  # mr4: shear
            # shear = np.random.uniform(-30, 30)
            shear = 30
            x_mr[index] = idg.apply_transform(x_mr[index], {"shear": shear})
        elif mr == 5:  # mr5: Elastic transformation
            # alpha = np.random.uniform(5, 12)
            # sigma = np.random.uniform(1.5, 2)
            alpha, sigma = 8, 1.7
            x_mr[index] = mrp.elastic_transform(x_mr[index], alpha, sigma)
        elif mr == 6:
            k = np.random.uniform = 0.7
            x_mr[index] = mrp.grey_transform(x_mr[index], k, 1 - k)
        elif mr == 7:
            x_mr[index] = mrp.pepper_noise(x_mr[index])

    return x_mr, y_org


def generate_mr_data(x_data, y_data, rate, mr):
    """
    the function to generate mr data
    :param x_data:相关的数据集
    :param y_data: 标签数据集
    :param rate: follow-up test case的占比
    :param mr: mr的指代编号
    :return: 返回处理过后的数据集
    :return: 返回处理过后的数据集
    """
    if rate == 1:
        (x_temp, y_temp) = mr_data_generate(x_data, y_data, mr, len(y_data))
        return x_temp, y_temp
    else:
        org_num = round(len(x_data) * (1 - rate))
        mr_num = len(x_data) - org_num
        # x_data, y_data = shuffle_data(x_data, y_data)

        (x_org, y_org) = (x_data[0:org_num], y_data[0:org_num])
        # 复制一份避免因为引用，而导致原数据被污染。
        (x_omr, y_omr) = (x_org.copy(), y_org.copy())
        if org_num >= mr_num:
            # 可以直接产生
            (x_mr, y_mr) = (x_omr.copy(), y_omr.copy())
            (x_temp, y_temp) = mr_data_generate(x_mr[0:mr_num], y_mr[0:mr_num], mr)
            x_mtg = np.concatenate((x_org, x_temp))
            y_mtg = np.concatenate((y_org, y_temp))
        else:
            x_mtg, y_mtg = x_org.copy(), y_org.copy()

            generate_quotient = mr_num // org_num
            generate_residue = mr_num % org_num

            # generate_quotient
            for _ in range(generate_quotient):  # 循环产生
                (x_mr, y_mr) = (x_omr.copy(), y_omr.copy())
                (x_mtg_sec, y_mtg_sec) = mr_data_generate(x_mr, y_mr, mr)
                x_mtg = np.concatenate((x_mtg, x_mtg_sec))
                y_mtg = np.concatenate((y_mtg, y_mtg_sec))
            # generate_residue
            (x_mr, y_mr) = (x_omr.copy(), y_omr.copy())
            (x_mtg_sec_res, y_mtg_sec_res) = mr_data_generate(x_mr, y_mr, mr)
            x_mtg = np.concatenate((x_mtg, x_mtg_sec_res[0:generate_residue]))
            y_mtg = np.concatenate((y_mtg, y_mtg_sec_res[0:generate_residue]))

            # # 产生余下的数据
            # left_num = mr_num - org_num * mo_rate
            # (x_mr, y_mr) = (x_org.copy(), y_org.copy())
            # (x_temp, y_temp) = mr_data_generate(x_mr, y_mr, mr, left_num)
            # x_omr = np.concatenate((x_omr, x_temp))
            # y_omr = np.concatenate((y_omr, y_temp))

        # elif generate_method == 'dg':  # 意味着不再加入MR，而是采用类似于数据增强的方式
        #     org_num = int(len(x_data) * (1 - rate))
        #     mr_num = len(x_data) - org_num
        #     (x_omr, y_omr) = (x_data[0:org_num].copy(), y_data[0:org_num].copy())
        #     x_data, y_data = shuffle_data(x_data, y_data)
        #     (x_temp, y_temp) = mr_data_generate(x_data, y_data, mr, mr_num)
        #     x_omr = np.concatenate((x_omr, x_temp))
        #     y_omr = np.concatenate((y_omr, y_temp))

        return x_mtg, y_mtg


def cal_metrics(x_mr, y_mr, model, mfr):
    # 检测能力
    # rov = rate_of_violation(y_mr, x_mr, model, rate)
    # 确定要求取那一层的输出
    model_input = model.layers[0].input
    model_output = [layer.output for layer in model.layers if
                    ('flatten' not in layer.name and 'input' not in layer.name)]
    get_layers_output = tf.keras.backend.function([model_input], model_output)
    # 必须分批提取参数，否则会严重溢出
    beg, end = 0, 10
    layers_output = None
    while end <= len(x_mr):
        layers_output_batch = get_layers_output(x_mr[beg:end])
        for layer_index in range(len(layers_output_batch)):
            if len(layers_output_batch[layer_index].shape) > 2:
                axis = tuple((i for i in range(1, len(layers_output_batch[layer_index].shape) - 1)))
                layers_output_batch[layer_index] = layers_output_batch[layer_index].mean(axis=axis)
        if layers_output is None:
            layers_output = layers_output_batch
        else:
            for layer_index in range(len(layers_output_batch)):
                layers_output[layer_index] = np.concatenate(
                    (layers_output[layer_index], layers_output_batch[layer_index]))
        beg += 10
        end += 10
        # print(end)
    # neuron level coverage
    MultiSecNeuCov, NeuBoundCov, StrNeuActCov = cov.cal_neulev_cov(layers_output, mfr)
    # layer level coverage
    TKNCov, TNKPat = cov.cal_laylev_cov(layers_output, mfr)
    return [MultiSecNeuCov, NeuBoundCov, StrNeuActCov, TKNCov, TNKPat]


def write_xlsx(file_path, list_input_data):
    wb = opxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    # # ws.append(
    #     ['model_name', 'size', 'rate of violation', 'MultiSecNeuCov',
    #      'NeuBoundCov', 'StrNeuActCov', 'TKNCov', 'TNKPat'])
    ws.append(list_input_data)
    wb.save(file_path)


def update_xlsx(file_path, list_input_data):
    if os.path.exists(file_path):
        wb = opxl.load_workbook(file_path)
        ws = wb['Sheet1']
        ws.append(list_input_data)
        wb.save(file_path)
    else:
        write_xlsx(file_path, list_input_data)
