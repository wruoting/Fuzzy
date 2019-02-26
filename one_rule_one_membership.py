import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


def trimf(x_len, abc):
    """
    Triangular membership function generator.

    Parameters
    ----------
    x_len : 1d array
        Independent variable.
    abc : 1d array, length 3
        Three-element vector controlling shape of triangular function.
        Requires a <= b <= c.

    Returns
    -------
    x: 1d array
        Triangular membership function.
    y : 1d array
        Triangular membership function.
    """
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = np.r_[abc]     # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = np.zeros(x_len)
    x = np.append(np.append(np.arange(a, b, np.divide(np.subtract(b, a), np.subtract(np.divide(x_len, 2), 1))),
                  np.arange(b, c, np.divide(np.subtract(c, b), np.divide(x_len, 2)))), c)

    # Left side
    if a != b:
        slope_a_b = np.divide(1, np.subtract(b, a))
        idx = np.nonzero(np.logical_and(a <= x, x <= b))
        y[idx] = np.multiply(slope_a_b, x[idx]-a)

    # Right side
    if b != c:
        slope_b_c = np.divide(1, np.subtract(c, b))
        idx = np.nonzero(np.logical_and(b < x, x <= c))
        y[idx] = np.multiply(slope_b_c, c - x[idx])

    # Middle is the mean
    idx = np.nonzero(x == b)
    y[idx] = 1
    return x, y

# A 1-d Vector of X data
def generate_mf(data):

    # Generate mean, min, max of universe
    mean_data = np.average(data)
    min_data = np.min(data)
    max_data = np.max(data)
    return trimf(len(data), [min_data, mean_data, max_data])


def generate_output(input_tag, output_tag, input_value, control, graph=False):
    control_simulation = ctrl.ControlSystemSimulation(control)

    # Compute an input to output
    control_simulation.input[input_tag] = input_value

    control_simulation.compute()
    if graph:
        fuzzy_output.view(sim=control_simulation)
    return control_simulation.output[output_tag]


# Generate some data; input and output
data_x = np.random.rand(10)
data_y = np.random.rand(10)

# Create an antecedent input set and a membership function
fuzzy_x, fuzzy_x_output = generate_mf(data_x)

fuzzy_input = ctrl.Antecedent(fuzzy_x, 'x')
fuzzy_input['x'] = fuzzy_x_output
fuzzy_input.view()

# Create a consequent output
fuzzy_y, fuzzy_y_output = generate_mf(data_y)
fuzzy_output = ctrl.Consequent(fuzzy_y, 'y')
fuzzy_output['y'] = fuzzy_y_output
fuzzy_output.view()

# Create a rule
rule1 = ctrl.Rule(fuzzy_input['x'], fuzzy_output['y'], label="rule")

# Create a control and controlsystem
control = ctrl.ControlSystem([rule1])


membership_output = []

# Store outputs to array
for datum in data_x:
    membership_output.append(generate_output('x', 'y', datum, control, graph=False))
mse_output = np.ones(len(membership_output))
mse = np.square(np.subtract(mse_output, membership_output))

print(mse)

