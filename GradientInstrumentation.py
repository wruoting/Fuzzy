import numpy as np
from FuzzySystem import FuzzySystem
from CreateSeedData import open_data, create_file
import matplotlib.pyplot as plt
from autograd import grad

def mse_generator(path=None):
    # Generate some data; input and output
    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))

    # Create our universe
    fuzzy_system = FuzzySystem(data_x, data_y)
    fuzzy_system.create_universes()

    # Create our MSE graph by creating a range of X's from min(data_x) to max(data_x)
    x_inputs = np.arange(np.min(fuzzy_system.data_x)+fuzzy_system.tol_x, np.max(fuzzy_system.data_x)-fuzzy_system.tol_x, fuzzy_system.tol_x)
    mse_array = []
    print('Creating MSEs')
    try:
        try_x, try_mse = open_data(path="{}normalized_peak_mse.txt".format(path))
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

        create_file(path="{}normalized_peak_mse.txt".format(path), x_data=x_inputs, y_data=mse_array)

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
    plt.savefig('{}data.png'.format(path))
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
    plt.savefig('{}overlay_data.png'.format(path))
    plt.close()


# Path Defaults
normalized_peak_path = "Data/NormalizedPeakCenter/"
normalized_peak_path_low_sample_size = "Data/NormalizedPeakCenterLowSampleSize/"
left_peak_path_low_sample_size = "Data/LeftPeakCenterLowSampleSize/"
left_shift_peak_path = "Data/LeftPeakCenter/"
right_shift_peak_path = "Data/RightPeakCenter/"
bimodal_peak_path = "Data/BimodalPeak/"
three_point_peak_path = "Data/ThreePointPeak/"

mse_generator(path=left_peak_path_low_sample_size)


def differentiate_fuzzy(x_value, fuzzy_system):
    grad_objective = grad(fuzzy_system.objective_function)
    return grad_objective(float(x_value))


def add_to_path(data_x, data_y, path):
    # Create our universe
    fuzzy_system = FuzzySystem(data_x, data_y)
    fuzzy_system.create_universes()
    min_x = np.min(data_x)
    max_x = np.max(data_x)
    x_array_linspace = np.linspace(max_x, min_x, 50)
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


def create_diff_data(path):
    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))
    add_to_path(data_x, data_y, path='{}DiffData_XY.txt'.format(path))


def plot_diff_data(path):
    data_x, data_y = open_data(path='{}DiffData_XY.txt'.format(path))
    plt.xlabel('X')
    plt.ylabel('dMSE/dX')
    plt.plot(data_x, data_y, 'ro')
    plt.savefig('{}dMSE_vs_dX_points.png'.format(path))
    plt.close()

    plt.xlabel('X')
    plt.ylabel('dMSE/dX')
    plt.plot(data_x, data_y)
    plt.savefig('{}dMSE_vs_dX.png'.format(path))
    plt.close()

    plt.show()


plot_diff_data(left_peak_path_low_sample_size)
