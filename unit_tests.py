from misc_functions import *
from CreateSeedData import *
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

three_point_peak_path_gauss = "Data/Non_Interpolated/ThreePointPeak/Gaussian_Data/"
path = three_point_peak_path_gauss
data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))
y_consequent = ctrl.Consequent(data_y, 'y')
granularity = 200
tol_y = np.divide(np.subtract(np.max(y_consequent.universe), np.min(y_consequent.universe)), granularity)

y_consequent_range = np.arange(np.min(y_consequent.universe) + tol_y,
                               np.max(y_consequent.universe) - tol_y, tol_y)

plt.plot(y_consequent_range, composite_gaussian(y_consequent_range, data_y, 2))
plt.show()
