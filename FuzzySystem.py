from ControlSystemSimulationOverride import ControlSystemSimulationOverride
from misc_functions import gaussian, gaussian_with_range
import numpy as np
import autograd.numpy as agnp
from skfuzzy import control as ctrl
import logging
logging.basicConfig(filename='variables.log', level=logging.DEBUG)


class FuzzySystem(object):

    def __init__(self, data_x, data_y, m_x=None, m_y=None,  analysis_function='gauss', path=None):
        self.data_x = data_x
        self.data_y = data_y
        self.min_x = np.min(self.data_x)
        self.max_x = np.max(self.data_x)
        self.min_y = np.min(self.data_y)
        self.max_y = np.max(self.data_y)
        self.tol_x = None
        self.tol_y = None
        self.x_antecedent = None
        self.y_consequent = None
        self.granularity = 200
        self.control = None
        self.rules = []
        self.control_simulation = None
        self.m_x = m_x if m_x else np.average(data_x)
        self.m_y = m_y if m_y else np.average(data_y)
        self.analysis_function = analysis_function
        self.analysis_params_antecedent = None
        self.analysis_params_consequent = None
        self.path = path
        self.std_dev_x = None
        self.std_dev_y = None
        self.std_x_sigma = None
        self.std_y_sigma = None

    def create_universes(self):
        # Set tolerance
        self.tol_x = np.divide(np.subtract(np.max(self.data_x), np.min(self.data_x)), self.granularity)
        self.tol_y = np.divide(np.subtract(np.max(self.data_y), np.min(self.data_y)), self.granularity)

        # Create an antecedent input set and a membership function
        self.x_antecedent = ctrl.Antecedent(self.data_x, 'x')

        # Create an consequent input set and a membership function
        self.y_consequent = ctrl.Consequent(self.data_y, 'y')

        self.std_dev_x = np.std(np.array(self.x_antecedent.universe))
        self.std_dev_y = np.std(np.array(self.y_consequent.universe))
        self.std_x_sigma = float(np.divide(self.std_dev_x, 6))
        self.std_y_sigma = float(self.std_dev_y)

    def create_membership(self, m_x=None, m_y=None):
        # so we don't actually use the antecedent and consequent stuff here...i might scrap all of this for a new repo
        if self.analysis_function == 'composite_gauss':
            # here we will create a composite gaussian of two gaussians, with mean E(x) and E(m_x) so we can adjust
            # our centroid as necessary
            if m_x:
                # our m_x gaussian has to fit within the range
                self.x_antecedent['x'] = gaussian(self.x_antecedent.universe, m_x,
                                                 float(np.std(np.array(self.x_antecedent.universe))))
                self.analysis_params_antecedent = {'mean': m_x,
                                                   'sigma': self.std_x_sigma,
                                                   'data': self.data_x,
                                                   'range': np.arange(np.min(self.data_x), np.max(self.data_x)+self.tol_x, self.tol_x),
                                                   'path': self.path}
            else:
                self.x_antecedent['x'] = gaussian(self.x_antecedent.universe,
                                                 float(np.mean(np.array(self.x_antecedent.universe))),
                                                 float(np.std(np.array(self.x_antecedent.universe))))
                self.analysis_params_antecedent = {'mean': float(np.mean(np.array(self.x_antecedent.universe))),
                                                   'sigma': float(np.std(np.array(self.x_antecedent.universe))),
                                                   'data': self.data_x,
                                                   'range': np.arange(np.min(self.data_x), np.max(self.data_x)+self.tol_x, self.tol_x),
                                                   'path': self.path}
            if m_y:
                # this is just a placeholder, self.y_consequent['y'] doesn't seem to affect results
                self.y_consequent['y'] = gaussian(self.y_consequent.universe,
                                                  float(np.mean(np.array(self.y_consequent.universe))),
                                                  self.std_y_sigma)
                self.analysis_params_consequent = {'mean': m_y,
                                                   'data': self.data_y,
                                                   'range': np.arange(np.min(self.data_y), np.max(self.data_y)+self.tol_y, self.tol_y),
                                                   'path': self.path}
            else:
                # this is just a placeholder, self.y_consequent['y'] doesn't seem to affect results
                self.y_consequent['y'] = gaussian(self.y_consequent.universe,
                                                  float(np.mean(np.array(self.y_consequent.universe))),
                                                  self.std_y_sigma)
                self.analysis_params_consequent = {'mean': float(np.mean(np.array(self.y_consequent.universe))),
                                                   'data': self.data_y,
                                                   'range': np.arange(np.min(self.data_y), np.max(self.data_y)+self.tol_y, self.tol_y),
                                                   'path': self.path}
        if self.analysis_function == 'gauss':

            if m_x:
                self.x_antecedent['x'] = gaussian(self.x_antecedent.universe, m_x,
                                                 float(np.std(np.array(self.x_antecedent.universe))))
                self.analysis_params_antecedent = {'mean': m_x,
                                                   'sigma': self.std_x_sigma,
                                                   'range': np.arange(np.min(self.data_x), np.max(self.data_x)+self.tol_x, self.tol_x),
                                                   'path': self.path}
            else:
                self.x_antecedent['x'] = gaussian(self.x_antecedent.universe,
                                                 float(np.mean(np.array(self.x_antecedent.universe))),
                                                 float(np.std(np.array(self.x_antecedent.universe))))
                self.analysis_params_antecedent = {'mean': float(np.mean(np.array(self.x_antecedent.universe))),
                                                   'sigma': float(np.std(np.array(self.x_antecedent.universe))),
                                                   'range': np.arange(np.min(self.data_x), np.max(self.data_x)+self.tol_x, self.tol_x),
                                                   'path': self.path}
            if m_y:
                self.y_consequent['y'] = gaussian(self.y_consequent.universe, m_y,
                                                 float(np.std(np.array(self.y_consequent.universe))))
                self.analysis_params_consequent = {'mean': m_y,
                                                   'sigma': float(np.std(np.array(self.y_consequent.universe))),
                                                   'range': np.arange(np.min(self.data_y), np.max(self.data_y)+self.tol_y, self.tol_y),
                                                   'path': self.path}
            else:
                self.y_consequent['y'] = gaussian(self.y_consequent.universe,
                                                  float(np.mean(np.array(self.y_consequent.universe))),
                                                  self.std_y_sigma)
                self.analysis_params_consequent = {'mean': float(np.mean(np.array(self.y_consequent.universe))),
                                                   'sigma': self.std_y_sigma,
                                                   'range': np.arange(np.min(self.data_y), np.max(self.data_y)+self.tol_y, self.tol_y),
                                                   'path': self.path}
        elif self.analysis_function == 'trimf':
            if m_x:
                self.x_antecedent['x'] = trimf(self.x_antecedent.universe,
                                               [np.min(self.data_x), m_x, np.max(self.data_x)])
            else:
                self.x_antecedent['x'] = trimf(self.x_antecedent.universe,
                                               [np.min(self.data_x), self.m_x, np.max(self.data_x)])
            if m_y:
                self.y_consequent['y'] = trimf(self.y_consequent.universe,
                                               [np.min(self.data_y), m_y, np.max(self.data_y)])
            else:
                self.y_consequent['y'] = trimf(self.y_consequent.universe,
                                               [np.min(self.data_y), self.m_y, np.max(self.data_y)])

    def rules_to_control(self):
        # Create a rule
        rule1 = ctrl.Rule(self.x_antecedent['x'], self.y_consequent['y'], label="rule1")

        self.rules = rule1
        # Create a control and controlsystem
        self.control = ctrl.ControlSystem(self.rules)
        self.control_simulation = ControlSystemSimulationOverride(self.control, self.analysis_function,
                                                                  self.analysis_params_antecedent,
                                                                  self.analysis_params_consequent)

    def objective_function(self, m_x):
        self.create_membership(m_x=m_x)
        self.rules_to_control()
        return self.mse

    def objective_function_middle_point(self, m_x):
        self.create_membership(m_x=m_x)
        self.rules_to_control()
        return self.single_point_mse

    def objective_function_membership(self, m_x):
        self.create_membership(m_x=m_x)
        self.rules_to_control()
        # Compute an input to output
        membership_output = []
        # Store outputs to array
        for datum in self.data_x:
            membership_output.append(self.generate_output('x', 'y', datum))
        return membership_output

    def generate_output(self, input_tag, output_tag, input_value):
        # Compute an input to output
        self.control_simulation.input[input_tag] = input_value
        try:
            self.control_simulation.compute()
        except ValueError:
            print('There was a value error generating this point')
            return 0

        return self.control_simulation.output[output_tag]

    @property
    def single_point_mse(self):
        middle_output = self.generate_output('x', 'y', self.data_x[1])
        mse = agnp.sum(np.square(agnp.subtract(self.data_y[1], middle_output)))
        return mse

    @property
    def mse(self):
        # Compute an input to output
        membership_output = []
        # Store outputs to array
        for datum in self.data_x:
            membership_output.append(self.generate_output('x', 'y', datum))
        mse = np.divide(np.sum(np.square(np.subtract(self.data_y, membership_output))), len(self.data_y))
        return mse

    def graph(self):
        self.x_antecedent.view()
        self.y_consequent.view()
        self.y_consequent.view(sim=self.control_simulation)

    def test_input(self, input_tag):
        choice = np.random.choice(self.data_x, 1)[0]
        choice_index = [index for index, value in enumerate(self.data_x) if value == choice]
        self.control_simulation.input[input_tag] = choice
        print('X data values: {}'.format(np.array2string(self.data_x)))
        print('Y data values: {}'.format(np.array2string(self.data_y)))
        print('Taking the {} value of X: '.format(choice_index))

        try:
            self.control_simulation.compute()
        except (ValueError, AssertionError):
            print('Defuzzification to 0')
            return 0
        self.x_antecedent.view()
        self.y_consequent.view(sim=self.control_simulation)


def trimf(x, abc):
    """
    Triangular membership function generator.

    Parameters
    ----------
    x : 1d array
        Independent variable.
    abc : 1d array, length 3
        Three-element vector controlling shape of triangular function.
        Requires a <= b <= c.

    Returns
    -------
    y : 1d array
        Triangular membership function.
    """
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = np.r_[abc]     # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = []
    temp_y = np.zeros(len(x))

    # Left side
    if a != b:
        idx_a_b = np.nonzero(np.logical_and(a <= x, x < b))[0]
        temp_y[idx_a_b] = -1
    if b != c:
        idx_b_c = np.nonzero(np.logical_and(b < x, x <= c))[0]
        temp_y[idx_b_c] = 1
    idx_b = np.nonzero(x == b)[0]
    temp_y[idx_b] = 0

    for index, value in enumerate(temp_y):
        if value == -1:
            y.append((x[index] - a) / (b - a))
        elif value == 1:
            y.append((c - x[index]) / (c - b))
        elif value == 0:
            y.append(1)
    return y


def gaussmf(x, mean, sigma):
    """
    Gaussian fuzzy membership function.

    Parameters
    ----------
    x : 1d array or iterable
        Independent variable.
    mean : float
        Gaussian parameter for center (mean) value.
    sigma : float
        Gaussian parameter for standard deviation.

    Returns
    -------
    y : 1d array
        Gaussian membership function for x.
    """
    return agnp.exp(-((x - mean)**2.) / (2 * sigma**2.))
