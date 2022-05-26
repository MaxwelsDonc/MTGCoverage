"""
main function to achieve our experiments
"""
import os

from tqdm.std import trange

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import para
import func
import numpy as np
import datetime
import time


def RQ1(model_name, data_name):
    date_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    par = para.Para()

    # if len(sys.argv) == 0:
    #     print('you are supposed to type the model name')
    #     sys.exit()
    # model_name = sys.argv[1]
    # gpu = sys.argv[2]
    # size = int(sys.argv[3])
    # mode = int(sys.argv[4])

    # model_name = 'LeNet1'
    gpu = '0'
    size = 1000
    # mode = int(sys.argv[4])

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"you have input the model_name {model_name}")
    print(f"you have set the gpu {gpu}")
    print(f"the size have been set to {size}")
    # model_name = 'ResNet50'
    model_data_dict = {'LeNet1': 'MNIST', 'LeNet4': 'MNIST', 'LeNet5': 'MNIST', 'VGG19': 'ImageNet',
                       'ResNet50': 'ImageNet'}

    # data_name = model_data_dict[model_name]

    # load the model and mgs
    time_start = time.time()
    print("start to load mgs")
    mgs_satisfied, mgs_violation = func.load_mgs(model_name, data_name)
    mgs_satisfied = func.shuffle_data(mgs_satisfied)
    mgs_violation = func.shuffle_data(mgs_violation)
    # mgs_satisfied_temp, mgs_violation_temp = mgs_satisfied[0:1000], mgs_violation[0:1000]
    # del mgs_violation, mgs_satisfied
    # mgs_satisfied, mgs_violation = mgs_satisfied_temp, mgs_violation_temp
    print("start to load model")
    model = func.load_model(model_name)
    print("start to load mfr")
    mfr = func.load_mfr(model_name)
    time_end = time.time()
    print(f"the total load time is {time_end - time_start}S")

    # we need adjust the violation rate of mgs
    temp = list(np.linspace(0, 1, 50))
    temp_list = [round(x, 2) for x in temp]
    # rate_list = temp_list[10 * mode:10 * mode + 10]
    rate_list = temp_list
    print(f"we will process the rate {rate_list[0]}--->{rate_list[-1]}")

    # generate the coverage dataset
    for rate in rate_list:
        print(f"model={model_name},size={size},rate={rate}")
        result = [model_name, size, rate]
        Metrics_list = []
        for run_index in range(30):
            # time_start = time.time()
            x_mgs, y_mgs = func.generate_xymgs(mgs_satisfied, mgs_violation, rate, size=size)
            Metrics = func.cal_metrics(x_mgs, y_mgs, model, mfr)
            Metrics_list.append(Metrics)
            # time_end = time.time()
        result.extend(list(np.average(Metrics_list, axis=0)))
        func.update_xlsx(f'../results/{model_name}{date_name}.xlsx', result)


def RQ2(model, mr_list):
    """
    因为实验数据的缘故，RQ2种只针对模型的部分MR数据进行了实验
    LeNet:1,2,3,4
    VGG19:1,4,6,7
    """
    date_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    par = para.Para()

    # if len(sys.argv) == 0:
    #     print('you are supposed to type the model name')
    #     sys.exit()
    # model_name = sys.argv[1]
    # gpu = sys.argv[2]
    # size = int(sys.argv[3])
    # mode = int(sys.argv[4])

    model_name = model
    gpu = '0'
    size = 1000
    # mode = int(sys.argv[4])

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"you have input the model_name {model_name}")
    print(f"you have set the gpu {gpu}")
    print(f"the size have been set to {size}")
    # model_name = 'ResNet50'
    model_data_dict = {'LeNet1': 'MNIST', 'LeNet4': 'MNIST', 'LeNet5': 'MNIST', 'VGG19': 'ImageNet',
                       'ResNet50': 'ImageNet'}

    data_name = model_data_dict[model_name]
    for mr in mr_list:
        # load the model and mgs
        time_start = time.time()
        print("start to load mgs")
        mgs_satisfied, mgs_violation = func.load_mgs(model_name, data_name, mr)
        print(f"the data mr={mgs_satisfied[1]['source'][-1]}")
        mgs_satisfied = func.shuffle_data(mgs_satisfied)
        mgs_violation = func.shuffle_data(mgs_violation)
        # mgs_satisfied_temp, mgs_violation_temp = mgs_satisfied[0:1000], mgs_violation[0:1000]
        # del mgs_violation, mgs_satisfied
        # mgs_satisfied, mgs_violation = mgs_satisfied_temp, mgs_violation_temp
        print("start to load model")
        model = func.load_model(model_name)
        print("start to load mfr")
        mfr = func.load_mfr(model_name)
        time_end = time.time()
        print(f"the total load time is {time_end - time_start}S")

        # we need adjust the violation rate of mgs
        temp = list(np.linspace(0, 1, 50))
        temp_list = [round(x, 2) for x in temp]
        # rate_list = temp_list[10 * mode:10 * mode + 10]
        rate_list = temp_list
        print(f"we will process the rate {rate_list[0]}--->{rate_list[-1]}")

        # generate the coverage dataset
        for rate in rate_list:
            print(f"model={model_name},size={size},rate={rate}")
            result = [model_name, size, rate]
            Metrics_list = []
            for run_index in trange(30):
                # time_start = time.time()
                x_mgs, y_mgs = func.generate_xymgs(mgs_satisfied, mgs_violation, rate, size=size)
                Metrics = func.cal_metrics(x_mgs, y_mgs, model, mfr)
                Metrics_list.append(Metrics)
                # time_end = time.time()
            result.extend(list(np.average(Metrics_list, axis=0)))
            # 如果不存在就进行初始化
            if not os.path.exists(f'../results/RQ2_{model_name}_{date_name}.xlsx'):
                func.update_xlsx(f'../results/RQ2_{model_name}_{date_name}.xlsx',
                                 ['model_name', 'size', 'rate of violation', 'MultiSecNeuCov', 'NeuBoundCov',
                                  'StrNeuActCov', 'TKNCov', 'TNKPat'])

            func.update_xlsx(f'../results/RQ2_{model_name}_{date_name}.xlsx', result)


if __name__ == '__main__':
    RQ1('LeNet1', 'fashionmnist')
