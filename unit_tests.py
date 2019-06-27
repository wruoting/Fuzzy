from misc_functions import *
from CreateSeedData import *

three_point_peak_path_gauss = "Data/Non_Interpolated/ThreePointPeak/Gaussian_Data/"
path = three_point_peak_path_gauss
data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))
composite_gaussian(data_x, [min(data_x), max(data_x)], tol=1e-6)