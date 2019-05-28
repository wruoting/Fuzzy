import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.defuzzify.defuzz import bisector
import autograd.numpy as agnp
from scipy.stats import norm
import scipy.integrate as integrate
from math import isclose

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


def gaussian(x, mean, sigma):
    sqrt_2pi = np.sqrt(np.multiply(2, np.pi))
    constant = np.divide(1, np.multiply(sigma, sqrt_2pi))
    exponent = -((x - mean)**2.) / (2 * sigma**2.)
    return np.multiply(constant, agnp.exp(exponent))


def centroid_gaussian(x, analysis_params):
    y_value = gaussian(x, analysis_params['mean'], analysis_params['sigma'])
    return x * y_value


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
            if isclose(y1, y2, abs_tol=1e-5):
                # we consider 5 digits to be close enough
                # this case is a square
                centroid_x.append(0.5*(x2**2-x1**2) * (x2-x1))
            else:
                # integrate from x1 to x2
                centroid_x.append(integrate.quad(lambda val: centroid_gaussian(val, analysis_params), x1, x2)[0]/integrate.quad(lambda val: gaussian(val, analysis_params['mean'], analysis_params['sigma']), x1, x2)[0])
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
