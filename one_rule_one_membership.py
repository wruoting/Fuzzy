import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# A 1-d Vector of X data
def generate_mf(data):

    # Generate mean, min, max of universe
    mean_data = np.average(data)
    min_data = np.min(data)
    max_data = np.max(data)

    return fuzz.trimf(data, [min_data, mean_data, max_data])


# Generate some data; input and output
data_x = np.array([0.1, 0.5, 1])
data_y = np.array([0.3, 0.7, 2])

# Create an antecedent input set and a membership function
fuzzy_input = ctrl.Antecedent(data_x, 'x')
fuzzy_input['x'] = generate_mf(data_x)
# fuzzy_input.view()

# Create a consequent output
fuzzy_output = ctrl.Consequent(data_y, 'y')
fuzzy_output['y'] = generate_mf(data_y)
# fuzzy_output.view()

# Create a rule
rule1 = ctrl.Rule(fuzzy_input['x'], fuzzy_output['y'], label="rule")

# Create a control and controlsystem
control = ctrl.ControlSystem([rule1])

control_simulation = ctrl.ControlSystemSimulation(control)

# Compute an input to output
control_simulation.input['x'] = 0.6

control_simulation.compute()
print(control_simulation.output['y'])
fuzzy_input.view(sim=control_simulation)
