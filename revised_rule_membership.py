import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def generate_output(input_tag, output_tag, input_value, control_simulation, y_consequent, graph=False):
    # Compute an input to output
    control_simulation.input[input_tag] = input_value
    try:
        control_simulation.compute()
    except ValueError:
        return 0
    if graph:
        y_consequent.view(sim=control_simulation)

    return control_simulation.output[output_tag]


def mse(data_x, data_y, control_simulation, y_consequent):
    # Compute an input to output
    membership_output = []

    # Store outputs to array
    for datum in data_x:
        membership_output.append(generate_output('x', 'y', datum, control_simulation, y_consequent))
    mse = np.sum(np.square(np.subtract(data_y, membership_output)))
    print(data_y)
    print(membership_output)
    print(mse)


def test_input(input_tag, x_antecedent, y_consequent, data_x, data_y):
    choice = np.random.choice(data_x, 1)[0]
    choice_index = [index for index, value in enumerate(data_x) if value == choice]
    control_simulation.input[input_tag] = choice
    print('X data values: {}'.format(np.array2string(data_x)))
    print('Y data values: {}'.format(np.array2string(data_y)))
    print('Taking the {} value of X: '.format(choice_index))

    try:
        control_simulation.compute()
    except (ValueError, AssertionError):
        print('Defuzzification to 0')
        return 0
    x_antecedent.view()
    y_consequent.view(sim=control_simulation)


# Generate some data; input and output
data_x = np.random.rand(10)
data_y = np.random.rand(10)

m_x = np.average(data_x)
m_y = np.average(data_y)

granularity = 500
tol_x = np.divide(np.subtract(np.max(data_x), np.min(data_x)), granularity)
tol_y = np.divide(np.subtract(np.max(data_y), np.min(data_y)), granularity)

# Create an antecedent input set and a membership function
x_antecedent = ctrl.Antecedent(np.arange(np.min(data_x), np.max(data_x), tol_x), 'x')
x_antecedent['x'] = fuzz.trimf(x_antecedent.universe, [np.min(data_x), m_x, np.max(data_x)])

# Create an consequent input set and a membership function
y_consequent = ctrl.Consequent(np.arange(np.min(data_y), np.max(data_y), tol_y), 'y')
y_consequent['y'] = fuzz.trimf(y_consequent.universe, [np.min(data_y), m_y, np.max(data_y)])

# Create a rule
rule1 = ctrl.Rule(x_antecedent['x'], y_consequent['y'], label="rule")

# Create a control and controlsystem
control = ctrl.ControlSystem([rule1])
control_simulation = ctrl.ControlSystemSimulation(control)

mse(data_x, data_y, control_simulation, y_consequent)

test_input('x', x_antecedent, y_consequent, data_x, data_y)
