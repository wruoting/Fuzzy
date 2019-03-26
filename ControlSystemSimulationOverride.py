from skfuzzy.control.controlsystem import ControlSystemSimulation, CrispValueCalculator
from misc_functions import interp_membership, defuzz, interp_universe_fast, centroid
import numpy as np


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

    def defuzz(self):
        """Derive crisp value based on membership of adjective(s)."""
        if not self.sim._array_inputs:
            ups_universe, output_mf, cut_mfs = self.find_memberships()

            if len(cut_mfs) == 0:
                raise ValueError("No terms have memberships.  Make sure you "
                                 "have at least one rule connected to this "
                                 "variable and have run the rules calculation.")

            try:
                return defuzz(ups_universe, output_mf,
                              self.var.defuzzify_method)
            except AssertionError:
                raise ValueError("Crisp output cannot be calculated, likely "
                                 "because the system is too sparse. Check to "
                                 "make sure this set of input values will "
                                 "activate at least one connected Term in each "
                                 "Antecedent via the current set of Rules.")
        else:
            # Calculate using array-aware version, one cut at a time.
            output = np.zeros(self.sim._array_shape, dtype=np.float64)

            it = np.nditer(output, ['multi_index'], [['writeonly', 'allocate']])

            for out in it:
                universe, mf = self.find_memberships_nd(it.multi_index)
                out[...] = defuzz(universe, mf, self.var.defuzzify_method)

            return output

    def find_memberships(self):
        '''
        First we have to upsample the universe of self.var in order to add the
        key points of the membership function based on the activation level
        for this consequent, using the interp_universe function, which
        interpolates the `xx` values in the universe such that its membership
        function value is the activation level.
        '''
        # Find potentially new values
        new_values = []

        for label, term in self.var.terms.items():
            term._cut = term.membership_value[self.sim]
            if term._cut is None:
                continue  # No membership defined for this adjective

            # Faster to aggregate as list w/duplication
            # self.var.universe - x's values
            # term.mf - y's values
            # term._cut - particular y value for area under

            new_values.extend(interp_universe_fast(self.var.universe, term.mf, term._cut))
        new_universe = np.union1d(self.var.universe, new_values)
        # Initialize membership
        output_mf = np.zeros_like(new_universe, dtype=np.float64)
        # Build output membership function
        term_mfs = {}
        upsampled_mf = []
        output_mf_final = []
        for label, term in self.var.terms.items():
            if term._cut is None:
                continue  # No membership defined for this adjective
            for value in new_universe:
                upsampled_mf.append(interp_membership(self.var.universe, term.mf, value))
            term_mfs[label] = np.minimum(term._cut, upsampled_mf)

            for output_mf_element, term_mf_element in zip(output_mf, term_mfs[label]):
                if output_mf_element >= term_mf_element:
                    output_mf_final.append(output_mf_element)
                elif output_mf_element < term_mf_element:
                    output_mf_final.append(term_mf_element)
        output_mf_final = np.array(output_mf_final)

        return new_universe, output_mf_final, term_mfs