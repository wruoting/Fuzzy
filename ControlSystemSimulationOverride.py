from skfuzzy.control.controlsystem import ControlSystemSimulation, CrispValueCalculator
from misc_functions import interp_membership


class ControlSystemSimulationOverride(ControlSystemSimulation):
    def __init__(self, control):
        super(ControlSystemSimulationOverride, self).__init__(control)

    def compute(self):
        """
        Compute the fuzzy system.
        """
        self.input._update_to_current()

        # Must clear downstream calculations for repeated runs
        if self._array_inputs:
            self.cache = False
            self._clear_outputs()

        # Shortcut with lookup if this calculation was done before
        if self.cache is not False and self.unique_id in self._calculated:
            for consequent in self.ctrl.consequents:
                self.output[consequent.label] = consequent.output[self]
            return

        # If we get here, cache is disabled OR the inputs are novel. Compute!

        # Check if any fuzzy variables lack input values and fuzzify inputs
        for antecedent in self.ctrl.antecedents:
            if antecedent.input[self] is None:
                raise ValueError("All antecedents must have input values!")
            CrispValueCalculatorOverride(antecedent, self).fuzz(antecedent.input[self])

        # Calculate rules, taking inputs and accumulating outputs
        first = True
        for rule in self.ctrl.rules:
            # Clear results of prior runs from Terms if needed.
            if first:
                for c in rule.consequent:
                    c.term.membership_value[self] = None
                    c.activation[self] = None
                first = False
            self.compute_rule(rule)

        # Collect the results and present them as a dict
        for consequent in self.ctrl.consequents:
            consequent.output[self] = \
                CrispValueCalculatorOverride(consequent, self).defuzz()
            self.output[consequent.label] = consequent.output[self]

        # Make note of this run so we can easily find it again
        if self.cache is not False:
            self._calculated.append(self.unique_id)
        else:
            # Reset StatePerSimulations
            self._reset_simulation()

        # Increment run number
        self._run += 1
        if self._run % self._flush_after_run == 0:
            self._reset_simulation()


class CrispValueCalculatorOverride(CrispValueCalculator):
    def __init__(self, fuzzy_var, sim):
        super(CrispValueCalculatorOverride, self).__init__(fuzzy_var, sim)

    def fuzz(self, value):
        """
        Propagate crisp value down to adjectives by calculating membership.
        """
        if len(self.var.terms) == 0:
            raise ValueError("Set Term membership function(s) first")

        for label, term in self.var.terms.items():
            term.membership_value[self.sim] = \
                interp_membership(self.var.universe, term.mf, value)
