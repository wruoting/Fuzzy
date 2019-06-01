import numpy as np
from FuzzySystem import FuzzySystem
from CreateSeedData import open_data, create_file, open_array_data
from misc_functions import gaussian
import matplotlib.pyplot as plt
from matplotlib import rcParams
from autograd import grad


def mse_generator(path=None, analysis_function='gauss'):
    # Generate some data; input and output
    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))

    mse_vs_x_graph = 'mse_vs_x.png'
    if analysis_function == 'gauss':
        normalized_peak_output_path = 'normalized_peak_mse_gauss.txt'
        normalized_peak_mse_output_path = 'normalized_peak_mse_gauss.txt'
        data_output_graph = 'data_gauss.png'
        overlay_graph_data = 'overlay_data_gauss.png'
    elif analysis_function == 'trimf':
        normalized_peak_output_path = 'normalized_peak_mse.txt'
        normalized_peak_mse_output_path = 'normalized_peak_mse.txt'
        data_output_graph = 'data.png'
        overlay_graph_data = 'overlay_data.png'

    # Create our universe
    fuzzy_system = FuzzySystem(data_x, data_y, analysis_function=analysis_function, path=path)
    fuzzy_system.create_universes()

    # Create our MSE graph by creating a range of X's from min(data_x) to max(data_x)
    mse_array = []
    print('Creating MSEs')
    try:
        try_x, try_mse = open_data(path="{}{}".format(path, normalized_peak_output_path))
        generate_data_mse = True
    except FileNotFoundError:
        generate_data_mse = False
        print("No File")

    if generate_data_mse:
        x_inputs = try_x
        mse_array = try_mse
    else:
        x_inputs = np.arange(np.min(fuzzy_system.data_x) + fuzzy_system.tol_x,
                             np.max(fuzzy_system.data_x) - fuzzy_system.tol_x, fuzzy_system.tol_x)
        for x_value in x_inputs:
            # Check the objective function here to make sure you're using the right one
            mse_array.append(fuzzy_system.objective_function_middle_point(m_x=x_value))
            print('Adding value for : {}'.format(x_value))
        create_file(path="{}{}".format(path, normalized_peak_mse_output_path), x_data=x_inputs, y_data=mse_array)

    plt.figure(0)
    plt.plot(x_inputs, mse_array)
    plt.xlabel('X values for membership peak')
    plt.ylabel('MSE of data set')
    plt.savefig('{}{}'.format(path, mse_vs_x_graph))
    plt.close()

    plt.figure(1)
    plt.plot(data_x, data_y, 'ro')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('{}{}'.format(path, data_output_graph))
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
    plt.savefig('{}{}'.format(path, overlay_graph_data))
    plt.close()


def centroid_generator(path=None, analysis_function='gauss'):
    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))

    if analysis_function == 'gauss':
        centroid_peak_output_path = 'centroid_peak_gauss.txt'
        centroid_peak_output_pic_path = 'centroid_data_gauss.png'
    centroid_y_array = []
    # Create our universe
    fuzzy_system = FuzzySystem(data_x, data_y, analysis_function=analysis_function, path=path)
    fuzzy_system.create_universes()
    # TODO: find a way to parse that file
    try:
        try_centroid_x, try_centroid_y = open_array_data(path="{}{}".format(path, centroid_peak_output_path))
        generate_data = True
    except FileNotFoundError:
        generate_data = False
        print("No File")
    if generate_data:
        centroid_x_array = try_centroid_x
        centroid_y_array = try_centroid_y
    else:
        x_inputs_centroid = np.arange(np.min(fuzzy_system.data_x) + fuzzy_system.tol_x,
                                      np.max(fuzzy_system.data_x) - fuzzy_system.tol_x, fuzzy_system.tol_x)
        centroid_x_array = []
        for x_value in x_inputs_centroid:
            centroid_x_array.append([data_x[0], data_x[1], data_x[2]])
            # centroid_x_array.append([x_value, x_value, x_value])
            centroid_y_array.append(fuzzy_system.objective_function_membership(m_x=x_value))
            print(fuzzy_system.objective_function_membership(m_x=x_value))
            print('Adding value for : {}'.format(x_value))
        create_file(path="{}{}".format(path, centroid_peak_output_path),
                    x_data=centroid_x_array, y_data=centroid_y_array)
    # centroid output
    plt.figure(2)
    for x_array, y_array, x_input in zip(centroid_x_array, centroid_y_array, x_inputs_centroid):
        x_input_path = '_' + str(x_input).replace('.', '_') + '_'
        granularity = 500
        tol_x = np.divide(np.subtract(np.max(data_x), np.min(data_x)), granularity)
        x_inputs_gaussian = np.arange(np.min(fuzzy_system.data_x) + tol_x, np.max(fuzzy_system.data_x) - tol_x, tol_x)
        x_outputs_gaussian = gaussian(x_inputs_gaussian, x_input, fuzzy_system.std_x_sigma)
        plt.plot(data_x, data_y, 'ro', label='Data Values')
        plt.plot(x_inputs_gaussian, x_outputs_gaussian, label='X-Membership Function Gaussian')
        plt.plot(x_array, y_array, label='Y-Membership Output Centroids')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.savefig('{}{}{}'.format(path, x_input_path, centroid_peak_output_pic_path))
        plt.close()


def differentiate_fuzzy(x_value, fuzzy_system):
    grad_objective = grad(fuzzy_system.objective_function)
    return grad_objective(float(x_value))


def graph_fuzzy(path, analysis_function='gauss'):
    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))
    fuzzy_system = FuzzySystem(data_x, data_y, analysis_function=analysis_function)
    fuzzy_system.create_universes()
    fuzzy_system.objective_function(5)
    fuzzy_system.graph()


def test_fuzzy(path, analysis_function='gauss'):
    data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))
    fuzzy_system = FuzzySystem(data_x, data_y, analysis_function=analysis_function)
    fuzzy_system.create_universes()
    fuzzy_system.create_membership(m_x=5)
    fuzzy_system.rules_to_control()
    for datum in data_x:
        print('X Value:')
        print(datum)
        print('Membership')
        print(fuzzy_system.generate_output('x', 'y', datum))

def add_to_path(data_x, data_y, path):
    # Create our universe
    fuzzy_system = FuzzySystem(data_x, data_y)
    fuzzy_system.create_universes()
    min_x = np.min(data_x)
    max_x = np.max(data_x)
    x_array_linspace = np.linspace(max_x, min_x, 200)
    try:
        open_data(path="{}".format(path))
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


def create_diff_data(path, analysis_function='gauss'):
    if analysis_function == 'gauss':
        normalized_peak_output_path = 'normalized_peak.txt'
        diff_data = 'DiffData_XY_Gauss.txt'
    elif analysis_function == 'trimf':
        normalized_peak_output_path = 'normalized_peak.txt'
        diff_data = 'DiffData_XY.txt'

    data_x, data_y = open_data(path="{}{}".format(path, normalized_peak_output_path))
    add_to_path(data_x, data_y, path='{}{}'.format(path, diff_data))


def plot_diff_data(path, analysis_function='gauss'):
    if analysis_function == 'gauss':
        output_dMSE_vs_dX_points = 'dMSE_vs_dX_points_gauss.png'
        output_dMSE_vs_dX = 'dMSE_vs_dX_gauss.png'
        diff_data = 'DiffData_XY_Gauss.txt'
        normalized_peak_output_path = 'normalized_peak_mse_gauss.txt'
        overlay_output_dMSE_vs_dX = 'overlay_dMSE_dX_gauss.png'
    elif analysis_function == 'trimf':
        output_dMSE_vs_dX_points = 'dMSE_vs_dX_points.png'
        output_dMSE_vs_dX = 'dMSE_vs_dX.png'
        diff_data = 'DiffData_XY.txt'
        normalized_peak_output_path = 'normalized_peak_mse.txt'
        overlay_output_dMSE_vs_dX = 'overlay_dMSE_dX.png'

    data_x_gauss, data_y_gauss = open_data(path='{}{}'.format(path, diff_data))
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    plt.xlabel('X')
    plt.ylabel('dMSE/dX')
    plt.plot(data_x_gauss, data_y_gauss, 'ro')
    plt.savefig('{}{}'.format(path, output_dMSE_vs_dX_points))
    plt.close()
    
    plt.tight_layout()
    plt.xlabel('X')
    plt.ylabel('dMSE/dX')
    plt.plot(data_x_gauss, data_y_gauss)
    plt.savefig('{}{}'.format(path, output_dMSE_vs_dX))
    plt.close()

    data_x_mse, data_y_mse = open_data(path="{}{}".format(path, normalized_peak_output_path))
    # Normalize Data overlay
    normalize_mse_y = normalize(data_y_mse, scaling_array=data_y_gauss)
    normalize_gauss_y = normalize(data_y_gauss, scaling_array=data_y_gauss)
    plt.plot(data_x_mse, normalize_mse_y, label="MSE")
    plt.plot(data_x_gauss, normalize_gauss_y, 'ro', label="dMSE/dX")
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('{}{}'.format(path, overlay_output_dMSE_vs_dX))
    plt.close()

    plt.show()


# From https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
def normalize(input_array, scaling_array=None):
    if scaling_array is not None:
        return np.multiply(np.divide(np.subtract(input_array, np.min(input_array)),
                                     np.subtract(np.max(input_array), np.min(input_array))),
                           np.subtract(np.max(scaling_array), np.min(scaling_array))) + np.min(scaling_array)
    return np.divide(np.subtract(input_array, np.min(input_array)), np.subtract(np.max(input_array), np.min(input_array)))


# Path Defaults
normalized_peak_path_low_sample_size = "Data/NormalizedPeakCenterLowSampleSize/Trim_ABC/"
normalized_peak_path_low_sample_size_gauss = "Data/NormalizedPeakCenterLowSampleSize/Gaussian_Data/"

left_peak_path_low_sample_size = "Data/LeftPeakCenterLowSampleSize/Trim_ABC/"
left_peak_path_low_sample_size_gauss = "Data/LeftPeakCenterLowSampleSize/Gaussian_Data/"

left_peak_path_low_sample_size_higher_sig = "Data/LeftPeakCenterHigherSigLowSampleSize/Trim_ABC/"
left_peak_path_low_sample_size_higher_sig_gauss = "Data/LeftPeakCenterHigherSigLowSampleSize/Gaussian_Data/"

left_shift_peak_path = "Data/LeftPeakCenter/Trim_ABC/"
left_shift_peak_path_gauss = "Data/LeftPeakCenter/Gaussian_Data/"

normalized_peak_path = "Data/NormalizedPeakCenter/Trim_ABC/"
normalized_peak_path_gauss = "Data/NormalizedPeakCenter/Gaussian_Data/"

right_shift_peak_path = "Data/RightPeakCenter/Trim_ABC/"
right_shift_peak_path_gauss = "Data/RightPeakCenter/Gaussian_Data/"

bimodal_peak_path = "Data/BimodalPeak/Trim_ABC/"
bimodal_peak_gauss = "Data/BimodalPeak/Gaussian_Data/"

three_point_peak_path = "Data/Non_Interpolated/ThreePointPeak/Trim_ABC/"
three_point_peak_path_gauss = "Data/Non_Interpolated/ThreePointPeak/Gaussian_Data/"

three_point_peak_right_path = "Data/ThreePointPeakRight/Trim_ABC/"
three_point_peak_right_path_gauss = "Data/ThreePointPeakRight/Gaussian_Data/"

three_point_peak_left_path = "Data/Non_Interpolated/ThreePointPeakLeft/Trim_ABC/"
three_point_peak_left_path_gauss = "Data/Non_Interpolated/ThreePointPeakLeft/Gaussian_Data/"

three_point_peak_left_increase_error_path = "Data/ThreePointPeakLeft_Increase_Error/Trim_ABC/"
three_point_peak_left_increase_error_path_gauss = "Data/ThreePointPeakLeft_Increase_Error/Gaussian_Data/"

three_point_peak_left_x_3_path = "Data/ThreePointPeakLeft_X_3/Trim_ABC/"
three_point_peak_left_x_3_path_gauss = "Data/ThreePointPeakLeft_X_3/Gaussian_Data/"

three_point_peak_left_x_4_path = "Data/ThreePointPeakLeft_X_4/Trim_ABC/"
three_point_peak_left_x_4_path_gauss = "Data/ThreePointPeakLeft_X_4/Gaussian_Data/"

three_point_peak_left_x_5_path = "Data/ThreePointPeakLeft_X_5/Trim_ABC/"
three_point_peak_left_x_5_path_gauss = "Data/ThreePointPeakLeft_X_5/Gaussian_Data/"

three_point_peak_left_x_5_400_pts_path = "Data/ThreePointPeakLeft_X_5_400_pts/Trim_ABC/"
three_point_peak_left_x_5_400_pts_path_gauss = "Data/ThreePointPeakLeft_X_5_400_pts/Gaussian_Data/"

mse_generator(path=three_point_peak_left_path_gauss, analysis_function='gauss')
# centroid_generator(path=three_point_peak_left_path_gauss, analysis_function='gauss')
# test_fuzzy(three_point_peak_left_path_gauss)
# create_diff_data(three_point_peak_left_path_gauss)
# plot_diff_data(path=three_point_peak_left_path_gauss)
# graph_fuzzy(path=three_point_peak_path_gauss)