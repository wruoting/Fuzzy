import numpy as np
from FuzzySystem import FuzzySystem
from CreateSeedData import open_data, create_file
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.preprocessing import MinMaxScaler


def mse_generator(path=None, function_type='gauss'):
    # Requires:
    # normalized_peak.txt
    # normalized_peak_mse_gauss.txt
    #
    # Output:
    # normalized_peak_mse_gauss.txt
    # data_gauss.png
    # overlay_data_gauss.png

    # Generate some data; input and output

    if function_type == 'gauss':
        open_path = 'normalized_peak_mse_gauss.txt'
        normalized_peak_mse_path = 'normalized_peak_mse_gauss.txt'
        fig_1_path = 'data_gauss.png'
        fig_2_path = 'overlay_data_gauss.png'
    elif function_type == 'trimf':
        open_path = 'normalized_peak_mse.txt'
        normalized_peak_mse_path = 'normalized_peak_mse.txt'
        fig_1_path = 'data.png'
        fig_2_path = 'overlay_data.png'

    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))

    # Create our universe
    fuzzy_system = FuzzySystem(data_x, data_y)
    fuzzy_system.create_universes()

    # Create our MSE graph by creating a range of X's from min(data_x) to max(data_x)
    x_inputs = np.arange(np.min(fuzzy_system.data_x)+fuzzy_system.tol_x, np.max(fuzzy_system.data_x)-fuzzy_system.tol_x, fuzzy_system.tol_x)
    mse_array = []
    print('Creating MSEs')
    try:
        try_x, try_mse = open_data(path="{}{}".format(path, open_path))
        generate_data = True
    except FileNotFoundError:
        generate_data = False
        print("No File")

    if generate_data:
        x_inputs = try_x
        mse_array = try_mse
    else:
        for x_value in x_inputs:
            mse_array.append(fuzzy_system.objective_function(m_x=x_value))
            print('Adding value for : {}'.format(x_value))
            create_file(path="{}{}".format(path, normalized_peak_mse_path), x_data=x_inputs, y_data=mse_array)

    plt.figure(0)
    plt.plot(x_inputs, mse_array)
    plt.xlabel('X values for membership peak')
    plt.ylabel('MSE of data set')
    plt.savefig('{}mse_vs_x.png'.format(path))
    plt.close()

    plt.figure(1)
    plt.plot(data_x, data_y, 'ro')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('{}{}'.format(path, fig_1_path))
    plt.close()

    # Normalize Data overlay
    normalize_mse = np.divide(np.subtract(mse_array, np.min(mse_array)), np.subtract(np.max(mse_array), np.min(mse_array)))
    normalize_y = np.divide(np.subtract(data_y, np.min(data_y)), np.subtract(np.max(data_y), np.min(data_y)))
    plt.figure(3)
    plt.plot(x_inputs, normalize_mse, label="MSE")
    plt.plot(data_x, normalize_y, 'ro', label="Data")
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('{}{}'.format(path, fig_2_path))
    plt.close()


def differentiate_fuzzy(x_value, fuzzy_system):
    grad_objective = grad(fuzzy_system.objective_function)
    return grad_objective(float(x_value))


def graph_fuzzy(path):
    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))
    fuzzy_system = FuzzySystem(data_x, data_y)
    fuzzy_system.create_universes()
    fuzzy_system.objective_function(0.1)
    fuzzy_system.graph()


def add_to_path(data_x, data_y, path):
    # Create our universe
    fuzzy_system = FuzzySystem(data_x, data_y)
    fuzzy_system.create_universes()
    min_x = np.min(data_x)
    max_x = np.max(data_x)
    x_array_linspace = np.linspace(max_x, min_x, 70)
    try:
        open_data(path=path)
        generate_data = False
    except FileNotFoundError:
        generate_data = True
        print("No File")
    if generate_data:
        f = open(path, 'w+')
        for value in x_array_linspace:
            f.write(str(value))
            f.write(" ")
        f.write(",")
        for index, value in enumerate(x_array_linspace):
            # if 43 >= index >= 37:
            f.write(str(differentiate_fuzzy(value, fuzzy_system)))
            f.write(" ")
            print("We are on point {}, index: {}".format(value, index))
        f.close()


def create_diff_data(path, function_type='gauss'):
    # Requires:
    # normalized_peak.txt
    #
    # Output:
    # DiffData_XY.txt
    if function_type == 'gauss':
        output_path = 'DiffData_XY_Gauss.txt'
    elif function_type == 'trimf':
        output_path = 'DiffData_XY.txt'

    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))
    add_to_path(data_x, data_y, path='{}{}'.format(path, output_path))


def plot_diff_data(path):
    # Requires:
    # DiffData_XY_Gauss.txt
    # normalized_peak_mse_gauss.txt
    #
    # Output:
    # dMSE_vs_dX_points_gauss.png
    # dMSE_vs_dX_gauss.png
    # overlay_dMSE_dX_gauss.png
    data_x, data_y = open_data(path='{}DiffData_XY_Gauss.txt'.format(path))
    plt.xlabel('X')
    plt.ylabel('dMSE/dX')
    plt.plot(data_x, data_y, 'ro')
    plt.savefig('{}dMSE_vs_dX_points_gauss.png'.format(path))
    plt.close()

    plt.xlabel('X')
    plt.ylabel('dMSE/dX')
    plt.plot(data_x, data_y)
    plt.savefig('{}dMSE_vs_dX_gauss.png'.format(path))
    plt.close()

    x_inputs, mse_array = open_data(path="{}normalized_peak_mse_gauss.txt".format(path))

    # normalize to y
    normalize_mse_y = normalize(mse_array, scale_to_array=data_y)
    normalize_y = data_y
    plt.plot(x_inputs, normalize_mse_y, label="MSE")
    plt.plot(data_x, normalize_y, 'ro', label="dMSE/dX")
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('{}overlay_dMSE_dX_gauss.png'.format(path))
    plt.close()

    plt.show()


def normalize(init_array, scale_to_array=None):
    if scale_to_array is not None:
        return np.multiply(np.divide(np.subtract(init_array, np.min(init_array)), np.subtract(np.max(init_array), np.min(init_array))),  np.subtract(np.max(scale_to_array), np.min(scale_to_array))) + np.min(scale_to_array)
    return np.divide(np.subtract(init_array, np.min(init_array)), np.subtract(np.max(init_array), np.min(init_array)))

# Path Defaults
normalized_peak_path = "Data/NormalizedPeakCenter/"
normalized_peak_path_low_sample_size = "Data/NormalizedPeakCenterLowSampleSize/Trim_ABC/"
normalized_peak_path_low_sample_size_gauss = "Data/NormalizedPeakCenterLowSampleSize/Gaussian_Data/"

left_peak_path_low_sample_size = "Data/LeftPeakCenterLowSampleSize/"
left_peak_path_low_sample_size_gauss = "Data/LeftPeakCenterLowSampleSize/Gaussian_Data/"

left_shift_peak_path = "Data/LeftPeakCenter/"
right_shift_peak_path = "Data/RightPeakCenter/"
bimodal_peak_path = "Data/BimodalPeak/"
three_point_peak_path = "Data/ThreePointPeak/"

# graph_fuzzy(path=left_peak_path_low_sample_size_gauss)
mse_generator(path=normalized_peak_path_low_sample_size_gauss)
# create_diff_data(left_peak_path_low_sample_size_gauss)
# plot_diff_data(path=left_peak_path_low_sample_size_gauss)
