import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from FuzzySystem import FuzzySystem
import matplotlib.pyplot as plt

# Generate some data; input and output
data_x = np.array([0.97587621, 0.18514317, 0.11278001, 0.59802743, 0.34573652, 0.65227173,
                   0.55377137, 0.85699145, 0.53695026, 0.26970461])
data_y = np.array([0.34010635, 0.27626549, 0.43559899, 0.25595622, 0.78174959, 0.60381853,
                   0.97123691, 0.83535587, 0.74909083, 0.4041416])

fuzzy_system = FuzzySystem(data_x, data_y)
fuzzy_system.create_universes()

x_inputs = np.arange(np.min(fuzzy_system.data_x)+fuzzy_system.tol_x, np.max(fuzzy_system.data_x)-fuzzy_system.tol_x, fuzzy_system.tol_x)
mse_array = []

for x_value in x_inputs:
    mse_array.append(fuzzy_system.objective_function(m_x=x_value))


plt.plot(x_inputs, mse_array)
plt.show()

for x, y in zip(x_inputs, mse_array):
    if y == np.min(mse_array):
        mse_x = x

fuzzy_system.objective_function(m_x=mse_x)
fuzzy_system.graph()



