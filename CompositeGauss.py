from misc_functions import *


class CompositeGauss(object):
    def __init__(self, universe, data_x, m_x):
        self.universe = universe
        self.new_universe = None
        self.data_x = data_x
        self.m_x = m_x
        self.mean_one = None
        self.mean_two = None
        self.sigma_one = None
        self.sigma_two = None

    def composite_gaussian(self):
        '''
        We will make two gaussians overlap
        Gaussian 1 will have a mean that's the mean of the data and a range that's the range of the data
        Gaussian 2 will have a mean that is either the lower or upper bound of the range around the mean of x
        :param universe: an array of x_values
        :param data_x: the data points
        :param m_x: the range of all the composite gaussians
        :return: composite gaussian x ranges and sigmas in a tuple
        The first peak will be the mean of the data, the second is dependent on the m_x that is provided
        '''

        # Check for x values to be within range x
        mean = np.mean(self.data_x)
        self.mean_one = mean
        # easy way to get the tolerance
        tol_universe = self.universe[1]-self.universe[0]
        # first gaussian
        revised_universe_range, sigma = gaussian_with_range(self.universe, np.mean(self.universe))

        self.sigma_one = sigma
        if self.m_x > mean:
            # our med is greater than the mean
            # We will use the right bound range as our new sigma
            second_mean = 2 * self.m_x - mean
            # If the second mean is greater than the max, we use the max
            if np.max(self.universe) < second_mean:
                second_mean = np.max(self.universe)
            # we are going to use 6 sigma
            second_sigma = np.divide((np.max(self.universe) - second_mean), 6)
        elif self.m_x < mean:
            # our med is less than the mean
            # We will use the left bound range as our new sigma
            second_mean = self.m_x - (mean - self.m_x)
            if np.min(self.universe) > second_mean:
                second_mean = np.min(self.universe)
            second_sigma = np.divide((second_mean - np.min(self.universe)), 6)
        else:
            # they are equal which means we don't have to do anything
            second_mean = mean
            second_sigma = np.divide((np.max(self.universe) - second_mean), 6)
        self.mean_two = second_mean
        self.sigma_two = second_sigma
        # tol total range / universe size
        if tol_universe < 6*sigma:
            second_universe = np.arange(second_mean - 6 * second_sigma, second_mean + 6 * second_sigma+tol_universe, tol_universe)
        else:
            raise Exception("Tolerance of this universe is greater than the spread")
        new_gaussian = np.array([])
        if second_sigma == 0:
            for x_value in np.arange(np.min(revised_universe_range), np.max(revised_universe_range),
                                     tol_universe):
                new_gaussian = np.append(new_gaussian, gaussian(x_value, mean, sigma))
            self.new_universe = np.arange(np.min(revised_universe_range), np.max(revised_universe_range), tol_universe)
        else:
            # We have a second gaussian to account for
            for x_value in np.arange(np.minimum(np.min(revised_universe_range), np.min(second_universe)),
                                     np.maximum(np.max(revised_universe_range), np.max(second_universe)),
                                     tol_universe):
                new_gaussian = np.append(new_gaussian, np.maximum(gaussian(x_value, mean, sigma), gaussian(x_value, second_mean, second_sigma)))
                self.new_universe = np.arange(np.minimum(np.min(revised_universe_range), np.min(second_universe)),
                                        np.maximum(np.max(revised_universe_range), np.max(second_universe)), tol_universe)
        normalize_new_gaussian = np.divide(np.subtract(new_gaussian, np.min(new_gaussian)), np.subtract(np.max(new_gaussian), np.min(new_gaussian)))
        
    
        return normalize_new_gaussian
