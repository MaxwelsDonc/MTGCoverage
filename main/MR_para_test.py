import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import para
import func
import numpy as np
import datetime

import time
import matplotlib.pyplot as plt

mr_list = [1, 2, 3, 4, 5, 6, 7]
# mr_list = [1]
mr_list_name = {0: 'Nochange', 1: 'Rotation', 2: 'Shift', 3: 'Zoom', 4: 'Shear', 5: 'Elastic',
                             6: 'Greychange', 7: 'Peppernoise'}
data_name = 'ImageNet'
(x_train, y_train), (x_test, y_test) = func.load_data(data_name)
for mr in mr_list:
    k = random.randint(0, len(y_test) - 1)
    x_org = x_test[k:k + 1].copy()
    x_follow, y_follow = func.mr_data_generate(x_org, y_test[k], mr)

    plt.subplot(1, 2, 1)
    plt.imshow(x_test[k]/255)
    plt.subplot(1, 2, 2)
    plt.imshow(x_follow[0]/255)
    plt.title(f"MR={mr_list_name[mr]}")
    plt.show()
