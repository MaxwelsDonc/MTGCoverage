"""
this script is used to store the parameters of our experiments
"""


class Para():
    """
    size_list(list): para storing the sizes of dataset
    rate_list(list): para storing the rate of f-u test cases in augmented dataset
    mr_list(list): para storing the index of MR, which can be found in mr_list-name
    mr_list_name(dict): para used for inquiry from mr_list
    dataset_par_list: the para of experimenal dataset
    run_times(int): the number how many program runs under different configurations
    exp_match(lsit->cell) storing all match of model and data
    k_section 覆盖率参数
    top_k 覆盖率参数
    """

    def __init__(self):
        self.size_list = [1000]
        self.rate_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.mr_list = [1, 2, 3, 4, 5, 6, 7]
        self.mr_list_name = {0: 'Nochange', 1: 'Rotation', 2: 'Shift', 3: 'Zoom', 4: 'Shear', 5: 'Elastic',
                             6: 'Greychange', 7: 'Peppernoise'}
        self.dataset_par_list = [(size, rate) for size in self.size_list for rate in self.rate_list]
        self.run_times = 100
        # self.exp_match = [('ResNet50', 'ImageNet'),('VGG19', 'ImageNet')]
        # self.exp_match = [('VGG19', 'ImageNet')]
        self.exp_match = [('LeNet1', 'MNIST'), ('LeNet4', 'MNIST'), ('LeNet5', 'MNIST')]
        self.k_section = 100
        self.top_k = 1
