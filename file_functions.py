import numpy as np
import matplotlib.pyplot as plt
from misc_functions import gaussian
import os


def str_replace(value):
    str_value = value.split('_')[0]
    int_value = int(str_value)
    return int_value


def save_fig(path=None, mean=None, sigma=None, x_data=None, y_data=None, y_value=None):
    try:
        path_list = os.listdir("{}/Membership".format(path))
        y_value = str(y_value).replace('.', '_')
        if len(path_list) == 0:
            next_number = '0_{}'.format(y_value)
        else:
            path_list = np.array([str_replace(value) for value in path_list])
            max_path = np.max(path_list) + 1
            next_number = str(max_path) + '_{}'.format(y_value)
        storage_fig_path = "{}/Membership/{}".format(path, next_number)
        gaussian_data_y = gaussian(x_data, mean, sigma)

        plt.plot(x_data, gaussian_data_y, label='Gaussian')
        plt.plot(x_data, y_data, label='Membership')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.savefig('{}'.format(storage_fig_path))
        plt.close()
    except Exception as e:
        print(e)
    return
