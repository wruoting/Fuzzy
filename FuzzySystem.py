import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzySystem(object):

    def __init__(self, data_x, data_y, m_x=None, m_y=None):
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
        self.granularity = 500
        self.control = None
        self.rules = []
        self.control_simulation = None
        self.m_x = m_x if m_x else np.average(data_x)
        self.m_y = m_y if m_y else np.average(data_y)

    def create_universes(self):
        # Set tolerance
        self.tol_x = np.divide(np.subtract(np.max(self.data_x), np.min(self.data_x)), self.granularity)
        self.tol_y = np.divide(np.subtract(np.max(self.data_y), np.min(self.data_y)), self.granularity)

        # Create an antecedent input set and a membership function
        self.x_antecedent = ctrl.Antecedent(np.arange(np.min(self.data_x), np.max(self.data_x), self.tol_x), 'x')

        # Create an consequent input set and a membership function
        self.y_consequent = ctrl.Consequent(np.arange(np.min(self.data_y), np.max(self.data_y), self.tol_y), 'y')

    def create_membership(self, m_x=None, m_y=None):
        if m_x:
            self.x_antecedent['x'] = fuzz.trimf(self.x_antecedent.universe,
                                                [np.min(self.data_x), m_x, np.max(self.data_x)])
        else:
            self.x_antecedent['x'] = fuzz.trimf(self.x_antecedent.universe,
                                                [np.min(self.data_x), self.m_x, np.max(self.data_x)])
        if m_y:
            self.y_consequent['y'] = fuzz.trimf(self.y_consequent.universe,
                                                [np.min(self.data_y), m_y, np.max(self.data_y)])
        else:
            self.y_consequent['y'] = fuzz.trimf(self.y_consequent.universe,
                                                [np.min(self.data_y), self.m_y, np.max(self.data_y)])

    def rules_to_control(self):
        # Create a rule
        rule1 = ctrl.Rule(self.x_antecedent['x'], self.y_consequent['y'], label="rule1")

        self.rules = rule1
        # Create a control and controlsystem
        self.control = ctrl.ControlSystem(self.rules)
        self.control_simulation = ctrl.ControlSystemSimulation(self.control)

    def objective_function(self, m_x):
        self.create_membership(m_x=m_x)
        self.rules_to_control()
        return self.mse

    def generate_output(self, input_tag, output_tag, input_value):
        # Compute an input to output
        self.control_simulation.input[input_tag] = input_value
        try:
            self.control_simulation.compute()
        except ValueError:
            return 0

        return self.control_simulation.output[output_tag]

    @property
    def mse(self):
        # Compute an input to output
        membership_output = []
        # Store outputs to array
        for datum in self.data_x:
            membership_output.append(self.generate_output('x', 'y', datum))
        mse = np.sum(np.square(np.subtract(self.data_y, membership_output)))
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




