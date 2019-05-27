import matplotlib.pyplot as plt
import numpy as np
from skfuzzy import control as ctrl
from misc_functions import gaussian, inverse_gaussian
from CreateSeedData import open_data
from misc_functions import skew_norm_pdf


def graph_gaussian(path=None):
    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))
    granularity = 500
    tol_x = np.divide(np.subtract(np.max(data_x), np.min(data_x)), granularity)
    new_universe = np.arange(np.min(data_x), np.max(data_x)+tol_x, tol_x)

    std_dev = np.std(data_x)
    result = gaussian(new_universe, 2, std_dev)
    result_skew = skew_norm_pdf(new_universe, e=2)
    normalize_result = np.divide(np.subtract(result, np.min(result)), np.subtract(np.max(result), np.min(result)))
    normalize_result_skew = np.divide(np.subtract(result_skew, np.min(result_skew)), np.subtract(np.max(result_skew), np.min(result_skew)))

    plt.plot(new_universe, normalize_result)
    plt.plot(new_universe, normalize_result_skew)
    # plt.show()


def graph_data(path=None):
    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))
    plt.xlabel('X')
    plt.ylabel('Y')
    normalize_y = np.divide(np.subtract(data_y, np.min(data_y)), np.subtract(np.max(data_y), np.min(data_y)))
    plt.plot(data_x, normalize_y, 'ro')
    # plt.show()


three_point_peak_left_path_gauss = "Data/Non_Interpolated/ThreePointPeakLeft/Gaussian_Data/"

graph_data(path=three_point_peak_left_path_gauss)
graph_gaussian(path=three_point_peak_left_path_gauss)
plt.show()
