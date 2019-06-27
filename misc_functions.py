import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.defuzzify.defuzz import bisector
import autograd.numpy as agnp
from scipy.stats import norm
import scipy.integrate as integrate
import scipy.special as special
from math import isclose
from scipy import optimize

def interp_membership(x, xmf, xx, tol=1e-5):
    """
    Find the degree of membership ``u(xx)`` for a given value of ``x = xx``.

    Parameters
    ----------
    x : 1d array
        Independent discrete variable vector.
    xmf : 1d array
        Fuzzy membership function for ``x``.  Same length as ``x``.
    xx : float or array of floats
        Value(s) on universe ``x`` where the interpolated membership is
        desired.
    zero_outside_x : bool, optional
        Defines the behavior if ``xx`` contains value(s) which are outside the
        universe range as defined by ``x``.  If `True` (default), all
        extrapolated values will be zero.  If `False`, the first or last value
        in ``x`` will be what is returned to the left or right of the range,
        respectively.

    Returns
    -------
    xxmf : float or array of floats
        Membership function value at ``xx``, ``u(xx)``.  If ``xx`` is a single
        value, this will be a single value; if it is an array or iterable the
        result will be returned as a NumPy array of like shape.

    -----
    For use in Fuzzy Logic, where an interpolated discrete membership function
    u(x) for discrete values of x on the universe of ``x`` is given. Then,
    consider a new value x = xx, which does not correspond to any discrete
    values of ``x``. This function computes the membership value ``u(xx)``
    corresponding to the value ``xx`` using linear interpolation.

    """
    # We overrode this function in an attempt to generate a line rather than an array

    # x is the input
    # xmf is the f(x)
    peak_y = np.max(xmf)
    peak_x = []
    for index, value in enumerate(xmf):
        if value == peak_y:
            peak_x = x[index]
    range_left = x[0]
    range_right = x[len(x)-1]
    slope_left = np.divide((peak_y - xmf[0]), (peak_x - range_left)) if peak_x != range_left else 0
    slope_right = np.divide((xmf[len(x)-1] - peak_y), (range_right - peak_x)) if peak_x != range_right else 0
    # plt.plot(x, xmf)
    if xx > peak_x:
        # plt.plot(xx, peak_y + slope_right * (xx - peak_x), marker="^")
        # plt.show()
        return peak_y+slope_right*(xx-peak_x) if peak_y+slope_right*(xx-peak_x) > tol else 0
    elif xx < peak_x:
        # plt.plot(xx, peak_y-slope_left*(peak_x-xx), marker="^")
        # plt.show()
        return peak_y-slope_left*(peak_x-xx) if peak_y-slope_left*(peak_x-xx) > tol else 0
    else:
        return peak_y if peak_y > tol else 0


def skew_norm_pdf(x, e=0, w=1, a=0):
    # adapated from:
    # http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy
    # e = location
    # w = scale
    # a = shape
    # mean = e + w*alpha*sqrt(2/pi)
    # alpha = alpha/sqrt(1+alpha^2)
    t = (x-e) / w
    return 2.0 * w * norm.pdf(t) * norm.cdf(a*t)


def inverse_skew_pdf(x, y, e=0, w=1, a=0):
    y_pdf = skew_norm_pdf(x, e=e, w=w, a=a)
    y_cdf = norm.cdf(y_pdf)
    return


def composite_gaussian(universe, m_x, tol=1e-6):
    '''
    We will make two gaussians overlap
    Gaussian 1 will have a mean that's the mean of the data and a range that's the range of the data
    Gaussian 2 will have a mean that is either the lower or upper bound of the range around the mean of x
    :param universe: an array of x_values
    :param m_x: the range of all the composite gaussians
    :param tol: tolerance of our range
    :return: composite gaussian x ranges and sigmas in a tuple
    '''

    # Check for x values to be within range x
    mean = np.mean(universe)
    # We need the size of the universe to get a 
    universe_size = np.size(universe)
    # first gaussian
    revised_universe_range, sigma = gaussian_with_range(universe, mean)
    first_gaussian = gaussian(revised_universe_range, mean, sigma)

    if m_x > mean:
        # our med is greater than the mean
        # We will use the right bound range as our new sigma
        second_mean = 2 * m_x - mean
        # If the second mean is greater than the max, we use the max
        if np.max(universe) < second_mean:
            second_mean = np.max(universe)
        # we are going to use 6 sigma
        second_sigma = np.divide((np.max(universe) - second_mean), 3)
    elif m_x < mean:
        # our med is less than the mean
        # We will use the left bound range as our new sigma
        second_mean = m_x - (mean - m_x)
        if np.min(universe) > second_mean:
            second_mean = np.max(universe)
        second_sigma = np.divide((second_mean - np.min(universe)), 3)
    else:
        # they are equal which means we don't have to do anything
        second_mean = mean
        second_sigma = np.divide((np.max(universe) - second_mean), 3)
    second_universe =
    second_gaussian = gaussian(revised_universe_range, second_mean, second_sigma)
    # normalize both to have a max of 1


def gaussian_with_range(universe, mean):
    """
    This function should always return a gaussian with the range of the initial universe
    However, it may not be centered at the center of that range.
    :param universe: np array that has the universe of points we are analysing in our fuzzy
    :param mean: the "mean" that you want to center your normal curve at
    :return: An antecedent range, the sigma of the gaussian
    """
    total_points = np.size(universe)
    total_range = np.max(universe) - np.min(universe)
    # sigma calculations are 6 sigma from the mean
    sigma = np.divide(np.sum(np.divide(total_range, 2), mean), 6)
    revised_universe_range = np.arange(mean - 6 * sigma,
                                       mean + 6 * sigma,
                                       np.divide(total_range, total_points))
    return revised_universe_range, sigma


def gaussian(x, mean, sigma):
    sqrt_2pi = np.sqrt(np.multiply(2, np.pi))
    constant = np.divide(1, np.multiply(sigma, sqrt_2pi))
    exponent = -((x - mean)**2.) / (2 * sigma**2.)
    return np.multiply(constant, agnp.exp(exponent))


def diff_gaussian(x, mean, sigma):
    exponent = -((x - mean)**2.) / (2 * sigma**2.)
    exp_term = agnp.exp(exponent)
    sqrt_2pi = np.sqrt(np.multiply(2, np.pi))
    sigma_3 = sqrt_2pi * sigma ** 3
    return np.divide((x-mean) * exp_term, sigma_3)


def centroid_gaussian(x, analysis_params):
    y_value = gaussian(x, analysis_params['mean'], analysis_params['sigma'])
    return x * y_value


# https://www.wolframalpha.com/input/?i=integrate+(1%2F((sigma)*sqrt(2*pi))*exp(-(x-mu)%5E2%2F(2*sigma%5E2)))%5E2+dx
def integrate_centroid(x, analysis_params):
    denom = 4*agnp.sqrt(np.pi) * analysis_params['sigma']
    return agnp.divide(special.erf((x-analysis_params['mean'])/analysis_params['sigma']), denom)


# https://www.wolframalpha.com/input/?i=integrate+(1%2F((sigma)*sqrt(2*pi))*exp(-(x-mu)%5E2%2F(2*sigma%5E2)))+dx
def integrate_gaussian(x, analysis_params):
    special_func = special.erf((x-analysis_params['mean'])/(agnp.sqrt(2) * analysis_params['sigma']))
    return agnp.divide(special_func, 2)


def inverse_gaussian(y, mean, sigma):
    sqrt_2pi = np.sqrt(np.multiply(2, np.pi))
    log_y = agnp.log(np.multiply(y, np.multiply(sigma, sqrt_2pi)))
    ln_calc = agnp.sqrt(np.multiply(-2, log_y))
    return np.add(np.multiply(sigma, ln_calc), mean)
    # we know we have two values, but either one should work for the fuzzification of our gaussian
    # return [np.add(np.multiply(sigma, ln_calc), mean), np.add(np.multiply(sigma, np.multiply(-1, ln_calc)), mean)]


def defuzz(x, mfx, mode, analysis_function, analysis_params):
    """
    Defuzzification of a membership function, returning a defuzzified value
    of the function at x, using various defuzzification methods.

    Parameters
    ----------
    x : 1d array or iterable, length N
        Independent variable.
    mfx : 1d array of iterable, length N
        Fuzzy membership function.
    mode : string
        Controls which defuzzification method will be used.
        * 'centroid': Centroid of area
        * 'bisector': bisector of area
        * 'mom'     : mean of maximum
        * 'som'     : min of maximum
        * 'lom'     : max of maximum

    Returns
    -------
    u : float or int
        Defuzzified result.

    See Also
    --------
    skfuzzy.defuzzify.centroid, skfuzzy.defuzzify.dcentroid
    """
    mode = mode.lower()
    x = x.ravel()
    mfx = mfx.ravel()
    n = len(x)
    assert n == len(mfx), 'Length of x and fuzzy membership function must be \
                          identical.'

    if 'centroid' in mode or 'bisector' in mode:
        zero_truth_degree = mfx.sum() == 0  # Approximation of total area
        assert not zero_truth_degree, 'Total area is zero in defuzzification!'

        if 'centroid' in mode:
            return centroid(x, mfx, analysis_function, analysis_params)

        elif 'bisector' in mode:
            return bisector(x, mfx)

    elif 'mom' in mode:
        return np.mean(x[mfx == mfx.max()])

    elif 'som' in mode:
        return np.min(x[mfx == mfx.max()])

    elif 'lom' in mode:
        return np.max(x[mfx == mfx.max()])

    else:
        raise ValueError('The input for `mode`, %s, was incorrect.' % (mode))


def interp_universe_fast(x, xmf, y):
    """
    Find interpolated universe value(s) for a given fuzzy membership value.

    Fast version, with possible duplication.

    Parameters
    ----------
    x : 1d array
        Independent discrete variable vector.
    xmf : 1d array
        Fuzzy membership function for ``x``.  Same length as ``x``.
    y : float
        Specific fuzzy membership value.

    Returns
    -------
    xx : list
        List of discrete singleton values on universe ``x`` whose
        membership function value is y, ``u(xx[i])==y``.
        If there are not points xx[i] such that ``u(xx[i])==y``
        it returns an empty list.

    Notes
    -----
    For use in Fuzzy Logic, where a membership function level ``y`` is given.
    Consider there is some value (or set of values) ``xx`` for which
    ``u(xx) == y`` is true, though ``xx`` may not correspond to any discrete
    values on ``x``. This function computes the value (or values) of ``xx``
    such that ``u(xx) == y`` using linear interpolation.
    """
    # Special case required or zero-level cut does not work with faster method
    if y == 0.:
        idx = np.where(np.diff(xmf > y))[0]
    else:
        idx = np.where(np.diff(xmf >= y))[0]

    # This method is fast, but duplicates point values where
    # y == peak of a membership function.

    # raise Exception
    return x[idx] + (y-xmf[idx]) * (x[idx+1]-x[idx]) / (xmf[idx+1]-xmf[idx])


def centroid(x, mfx, analysis_function, analysis_params):
    """
    Defuzzification using centroid (`center of gravity`) method.

    Parameters
    ----------
    x : 1d array, length M
        Independent variable
    mfx : 1d array, length M
        Fuzzy membership function

    Returns
    -------
    u : 1d array, length M
        Defuzzified result

    See also
    --------
    skfuzzy.defuzzify.defuzz, skfuzzy.defuzzify.dcentroid
    """

    '''
    As we suppose linearity between each pair of points of x, we can calculate
    the exact area of the figure (a triangle or a rectangle).
    '''

    sum_moment_area = 0.0
    sum_area = 0.0
    if analysis_function == 'gauss':
        # detect tol is flat
        centroid_x = []
        for i in range(1, len(x)):
            x1 = x[i - 1]
            x2 = x[i]
            y1 = mfx[i - 1]
            y2 = mfx[i]
            # if agnp.absolute(agnp.subtract(y1, y2)) <= 1e-5:
            #     # we consider 5 digits to be close enough
            #     # this case is a square
            #     centroid_x.append(0.5*(x2**2-x1**2) * (x2-x1))
            # else:
            # # integrate from x1 to x2
            # first_integration = integrate.quad(lambda val: centroid_gaussian(val, analysis_params), x1, x2)[0]
            # second_integration = integrate.quad(lambda val: gaussian(val, analysis_params['mean'], analysis_params['sigma']), x1, x2)[0]
            # total_integration = agnp.divide(first_integration, second_integration)
            if y2 == y1:
                centroid_x.append(0.5 * (x2 ** 2 - x1 ** 2) * (x2 - x1))
            elif y2 > y1:
                first_integration = integrate_centroid(x2, analysis_params) - integrate_centroid(x1, analysis_params)
                second_integration = integrate_gaussian(x2, analysis_params) - integrate_gaussian(x1, analysis_params)
                total_integration = agnp.divide(first_integration, second_integration)
                centroid_x.append(total_integration)
            elif y2 < y1:
                first_integration = integrate_centroid(x1, analysis_params) - integrate_centroid(x2, analysis_params)
                second_integration = integrate_gaussian(x1, analysis_params) - integrate_gaussian(x2, analysis_params)
                total_integration = agnp.divide(first_integration, second_integration)
                centroid_x.append(total_integration)
        if len(centroid_x) == 0:
            total_centroid = 0
        else:
            total_centroid = np.average(centroid_x)
        return total_centroid
    else:
        # If the membership function is a singleton fuzzy set:
        if len(x) == 1:
            print(x[0]*mfx[0] / np.fmax(mfx[0], np.finfo(float).eps).astype(float))
            return x[0]*mfx[0] / np.fmax(mfx[0], np.finfo(float).eps).astype(float)

        # else return the sum of moment*area/sum of area
        for i in range(1, len(x)):
            x1 = x[i - 1]
            x2 = x[i]
            y1 = mfx[i - 1]
            y2 = mfx[i]

            # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
            if not(y1 == y2 == 0.0 or x1 == x2):
                if y1 == y2:  # rectangle
                    moment = 0.5 * (x1 + x2)
                    area = (x2 - x1) * y1
                elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                    moment = 2.0 / 3.0 * (x2-x1) + x1
                    area = 0.5 * (x2 - x1) * y2
                elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                    moment = 1.0 / 3.0 * (x2 - x1) + x1
                    area = 0.5 * (x2 - x1) * y1
                else:
                    moment = (2.0 / 3.0 * (x2-x1) * (y2 + 0.5*y1)) / (y1+y2) + x1
                    area = 0.5 * (x2 - x1) * (y1 + y2)

                sum_moment_area += moment * area
                sum_area += area
        float_epsilon = 2.220446049250313e-16
        return sum_moment_area / np.fmax(sum_area, float_epsilon)
