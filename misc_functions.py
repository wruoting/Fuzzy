import numpy as np
import matplotlib.pyplot as plt


def interp_membership(x, xmf, xx):
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
    slope_left = (peak_y - xmf[0])/(peak_x - range_left)
    slope_right = (xmf[len(x)-1] - peak_y)/(range_right - peak_x)
    # plt.plot(x, xmf)
    if xx > peak_x:
        # plt.plot(xx, peak_y + slope_right * (xx - peak_x), marker="^")
        # plt.show()
        return peak_y+slope_right*(xx-peak_x)
    elif xx < peak_x:
        # plt.plot(xx, peak_y-slope_left*(peak_x-xx), marker="^")
        # plt.show()
        return peak_y-slope_left*(peak_x-xx)
    else:
        return peak_y
