# 本程序是用来进行MG的整体划分来进行实验的
import os
import sys
import func
import numpy as np
from tqdm import tqdm
from tqdm.std import trange

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 载入数据
# experiment_model_data_list = [('LeNet1', 'MNIST'), ('LeNet4', 'MNIST'), ('LeNet5', 'MNIST'), ('ResNet50', 'ImageNet'),
#                               ('VGG19', 'ImageNet')]

def separate_satisfied_violation(model_name, data_name):
    # if len(sys.argv) == 0:
    #     print('you are supposed to type the model name')
    #     sys.exit()
    # model_name = sys.argv[1]
    # print(f"you have input the model_name {model_name}")
    # model_data_dict = {'LeNet1': 'MNIST', 'LeNet4': 'MNIST', 'LeNet5': 'MNIST', 'VGG19': 'ImageNet',
    #                    'ResNet50': 'ImageNet'}
    # data_name = model_data_dict[model_name]
    mr_list = [1, 2, 3, 4, 6, 7]

    (_, _), (x_source, y_source) = func.load_data(data_name)
    model = func.load_model(model_name)
    # 产生MG
    # MG分类
    MGs_satisfied = []
    MGs_violation = []
    for mr in mr_list:
        print(f"we now are conducting experiment on MR{mr}:", end="")
        x_org, y_org = x_source.copy(), y_source.copy()
        x_follow, y_follow = func.mr_data_generate(x_org=x_org, y_org=y_org, mr=mr)

        for index in trange(len(y_source)):
            y_pre_source = np.argmax(model.predict(x_source[index:index + 1]), axis=-1)
            y_pre_follow = np.argmax(model.predict(x_follow[index:index + 1]), axis=-1)
            if y_pre_source == y_pre_follow:
                MG_satisfied = {}
                MG_satisfied['source'] = (x_source[index], y_source[index], y_pre_source, mr)
                MG_satisfied['follow'] = (x_follow[index], y_follow[index], y_pre_follow, mr)
                MGs_satisfied.append(MG_satisfied)
            else:
                MG_violation = {}
                MG_violation['source'] = (x_source[index], y_source[index], y_pre_source, mr)
                MG_violation['follow'] = (x_follow[index], y_follow[index], y_pre_follow, mr)
                MGs_violation.append(MG_violation)

    # 分别保存数据
    np.save(f'../data/{model_name}/MGs_satisfied{data_name}.npy', MGs_satisfied)
    np.save(f'../data/{model_name}/MGs_violation{data_name}.npy', MGs_violation)


def random_select_satisfied_violation(model_name, size):
    # load the mgs
    print("load the mgs")
    mgs_satisfied, mgs_violation = func.load_mgs(model_name)
    # shuffle the data
    print("mgs load complete, now begain to shuffle the data")
    mgs_satisfied = func.shuffle_data(mgs_satisfied, size=size)
    mgs_violation = func.shuffle_data(mgs_violation, size=size)
    print("save the data")
    np.save(f'../data/{model_name}/mgs_satisfied_size.npy', mgs_satisfied)
    np.save(f'../data/{model_name}/mgs_violation_size.npy', mgs_violation)
    print("all data are saved")


def separate_satisfied_violation_by_mr(model_name):
    # load the mgs
    print("load the mgs")
    mgs_satisfied, mgs_violation = func.load_mgs(model_name)
    # 进行分开存储
    mgs_v_sort_by_mr = {}
    mgs_s_sort_by_mr = {}
    # 遍历第一个变量进行分类
    print("process the mgs_satisfied")
    for mg_satisfied in mgs_satisfied:
        # 提取MR
        mr = mg_satisfied['source'][-1]
        # 如果这个MR对应的已经初始化过就直接加入
        if mr in mgs_s_sort_by_mr.keys():
            mgs_s_sort_by_mr[mr].append(mg_satisfied)
        # 如果没有的话,进行初始化
        else:
            mgs_s_sort_by_mr[mr] = []
            mgs_s_sort_by_mr[mr].append(mg_satisfied)

    # 遍历第二个变量
    print("process the mgs_violation")
    for mg_violation in mgs_violation:
        # 提取MR
        mr = mg_violation['source'][-1]
        # 如果这个MR对应的已经初始化过就直接加入
        if mr in mgs_v_sort_by_mr.keys():
            mgs_v_sort_by_mr[mr].append(mg_violation)
        # 如果没有的话,进行初始化
        else:
            mgs_v_sort_by_mr[mr] = []
            mgs_v_sort_by_mr[mr].append(mg_violation)

    # 变量太多用文件夹保存
    print("save the data")
    if not os.path.exists(f'../data/{model_name}/mgs_sv_mr'):
        print("generate the dir")
        os.mkdir(f'../data/{model_name}/mgs_sv_mr')
    # 开始存储
    for mr in list(mgs_v_sort_by_mr.keys()):
        np.save(f'../data/{model_name}/mgs_sv_mr/mgs_s_mr{mr}.npy', mgs_s_sort_by_mr[mr])
        np.save(f'../data/{model_name}/mgs_sv_mr/mgs_v_mr{mr}.npy', mgs_v_sort_by_mr[mr])
    print("save complete")
    # 存储相关的变量数据
    if not os.path.exists(f'../data/vs_info.xlsx'):
        var = ['model_name', '#MGs_satisfication', '#MGs_violation',
               '#MGs_s_MR1', '#MGs_s_MR2', '#MGs_s_MR3', '#MGs_s_MR4', '#MGs_s_MR5', '#MGs_s_MR6','#MGs_s_MR7',
               '#MGs_v_MR1', '#MGs_v_MR2', '#MGs_v_MR3', '#MGs_v_MR4', '#MGs_v_MR5', '#MGs_v_MR6','#MGs_v_MR7']
        func.update_xlsx(f'../data/vs_info.xlsx', var)

    var = [model_name, len(mgs_satisfied), len(mgs_violation),
           len(mgs_s_sort_by_mr[1]), len(mgs_s_sort_by_mr[2]), len(mgs_s_sort_by_mr[3]), len(mgs_s_sort_by_mr[4]),
           len(mgs_s_sort_by_mr[5]), len(mgs_s_sort_by_mr[6]),len(mgs_s_sort_by_mr[7]),
           len(mgs_v_sort_by_mr[1]), len(mgs_v_sort_by_mr[2]), len(mgs_v_sort_by_mr[3]), len(mgs_v_sort_by_mr[4]),
           len(mgs_v_sort_by_mr[5]), len(mgs_v_sort_by_mr[6]),len(mgs_v_sort_by_mr[7])]
    func.update_xlsx(f'../data/vs_info.xlsx', var)


if __name__ == '__main__':
    separate_satisfied_violation('LeNet1', 'fashionminist')
