import numpy as np
import matplotlib.pyplot as plt
from misc_functions import gaussian
import uuid


def save_fig(path=None, mean=None, sigma=None, x_data=None, y_data=None):
    filename = uuid.uuid4().hex
    storage_fig_path = "{}/Membership/{}".format(path, filename)
    gaussian_data_y = gaussian(x_data, mean, sigma)

    plt.plot(x_data, gaussian_data_y, label='Gaussian')
    plt.plot(x_data, y_data, label='Membership')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('{}'.format(storage_fig_path))
    plt.close()
    return
