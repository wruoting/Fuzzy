import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy.membership import generatemf as mf

quality = ctrl.Antecedent(np.arange(0, 26, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# Auto-membership function population is possible with .automf(3, 5, or 7)
service.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# You can see how these look with .view()
# These have to be run in an ipython terminal via %run
quality['high'] = mf.trimf(quality.universe, [13, 25, 25])
# We care about how to generate that triangular membership function
quality.view()



