from CreateSeedData import *
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from CompositeGauss import CompositeGauss
import numpy as np

three_point_peak_path_gauss = "Data/Non_Interpolated/ThreePointPeak/Gaussian_Data/"
path = three_point_peak_path_gauss
data_x, data_y = open_data(path="{}normalized_peak.txt".format(path))
y_consequent = ctrl.Consequent(data_y, 'y')
granularity = 300
tol_y = np.divide(np.subtract(np.max(y_consequent.universe), np.min(y_consequent.universe)), granularity)
y_consequent_range = np.arange(np.min(y_consequent.universe),
                               np.max(y_consequent.universe) + tol_y, tol_y)

cg = CompositeGauss(y_consequent_range, data_y, 3)
cg.composite_gaussian()
plt.plot(cg.new_universe, cg.composite_gaussian())
plt.plot(np.full(6, cg.mean_one), np.arange(0, 1.2, 0.2))
plt.plot(np.full(6, cg.mean_two), np.arange(0, 1.2, 0.2))
plt.show()
print(cg.sigma_one)
print(cg.sigma_two)
