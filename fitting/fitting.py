"""
Various functions which can be fitted to given data.

:requires: NumPy
:requires: SciPy

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk

:version: 0.1
"""
import scipy
import scipy.optimize
import numpy as np


def chi2(data, predicted, uncertainties):
    """
    Classical chi**2 goodness-of-fit estimator.

    :param data: input data
    :param predicted: predicted values
    :param uncertainties: errors of the data

    :return: chi**2
    """
    return np.sqrt(np.sum(((data-predicted)/uncertainties)**2,axis=-1))


def polynomial5(x, p0, p1, p2, p3, p4):
    """
    5th order polynomial
    """
    return p0 * x + p1 * x ** 2 + p2 * x ** 3 + p3 * x ** 4 + p4 * x ** 5


def GaussianSumsExample():
    """
    A simple example showing how to fit a sum of Gaussians.

    :return:
    """
    import matplotlib.pyplot as plt
    import scipy.stats

    data = np.array([-69, 3, -68, 2, -67, 1, -66, 1, -60, 1, -59, 1,
                     -58, 1, -57, 2, -56, 1, -55, 1, -54, 1, -52, 1,
                     -50, 2, -48, 3, -47, 1, -46, 3, -45, 1, -43, 1,
                     0, 1, 1, 2, 2, 12, 3, 18, 4, 18, 5, 13, 6, 9,
                     7, 7, 8, 5, 9, 3, 10, 1, 13, 2, 14, 3, 15, 2,
                     16, 2, 17, 2, 18, 2, 19, 2, 20, 2, 21, 3, 22, 1,
                     24, 1, 25, 1, 26, 1, 28, 2, 31, 1, 38, 1, 40, 2])
    x, y = data.reshape(-1, 2).T

    def tri_norm(x, *args):
        m1, m2, m3, s1, s2, s3, k1, k2, k3 = args
        ret = k1 * scipy.stats.norm.pdf(x, loc=m1, scale=s1)
        ret += k2 * scipy.stats.norm.pdf(x, loc=m2, scale=s2)
        ret += k3 * scipy.stats.norm.pdf(x, loc=m3, scale=s3)
        return ret


    params = [-50, 3, 20, 1, 1, 1, 1, 1, 1]

    fitted_params, _ = scipy.optimize.curve_fit(tri_norm, x, y, p0=params)

    plt.plot(x, y, 'o')
    xx = np.linspace(np.min(x), np.max(x), 1000)
    plt.plot(xx, tri_norm(xx, *fitted_params))
    plt.show()


def _gaussianMoments2D(data):
    """
    :return: (height, x, y, width_x, width_y) the gaussian parameters of a 2D distribution
             by calculating its moments
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    widthX = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    widthY = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, widthX, widthY


def _gaussian2D(height, center_x, center_y, width_x, width_y):
    """
    Returns a 2 dimensional gaussian function with the given parameters.
    """
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)


def Gaussian2D(data):
    """
    Fits a 2D Gaussian to a given data.
    Uses scipy.optimize.leastsq for fitting.

    :param data: data
    """
    params = _gaussianMoments2D(data)
    errfunc = lambda p: np.ravel(_gaussian2D(*p)(*np.indices(data.shape)) - data)
    p, success = scipy.optimize.leastsq(errfunc, params)
    return p


def Gaussian(ydata, xdata=None, initials=None):
    """
    Fits a single Gaussian to a given data.
    Uses scipy.optimize.leastsq for fitting.

    :param ydata: to which a Gaussian will be fitted to.
    :param xdata: if not given uses np.arange
    :param initials: initial guess for Gaussian parameters in order [amplitude, mean, sigma]

    :return: coefficients, best fit params, success
    :rtype: dictionary
    """
    # define a gaussian fitting function where
    # p[0] = amplitude
    # p[1] = mean
    # p[2] = sigma
    fitfunc = lambda p, x: p[0] * scipy.exp(-(x - p[1]) ** 2 / (2.0 * p[2] ** 2))
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    if initials is None:
        initials = scipy.c_[np.max(ydata), np.argmax(ydata), 5][0]

    if xdata is None:
        xdata = np.arange(len(ydata))

    # fit a gaussian to the correlation function
    p1, success = scipy.optimize.leastsq(errfunc, initials[:],\
        args=(xdata, ydata))

    # compute the best fit function from the best fit parameters
    corrfit = fitfunc(p1, xdata)

    out = {}
    out['fit'] = corrfit
    out['parameters'] = p1
    out['amplitude'] = p1[0]
    out['mean'] = p1[1]
    out['sigma'] = p1[2]
    out['fwhm'] = 2 * np.sqrt(2 * np.log(2)) * out['sigma']
    out['success'] = success

    return out


def linearregression(x, y,
                     confidence_level=95,
                     show_plots=False,
                     report=False):
    """
    A function to calculate simple linear regression
    inputs: x,y-data pairs and a confidence level (in percent)
    outputs: slope = b1, intercept = b0 its confidence intervals
    and R (+squared)
    """
    #Basic statistics
    x_average = np.mean(x)
    x_stdev = np.std(x)
    y_average = np.mean(y)
    y_stdev = np.std(y)
    n = len(x)

    if n == 2:
        #print 'Only 2 degrees of freedom, cannot calculate errors...'
        return 0, 0, 0

    # calculate linear regression coefficients
    b1 = np.sum((x - x_average) * (y - y_average)) / np.sum((x - x_average) ** 2)
    b0 = y_average - b1 * x_average

    # calculate residuals (observed - predicted)
    TotSS = np.sum((y - y_average) ** 2)   # Total Sum of Squares
    y_hat = b1 * x + b0               # fitted values
    ResSS = np.sum((y - y_hat) ** 2)       # Residual Sum of Squares

    # calculate standard deviations of fit params
    b1_stdev = np.sqrt((ResSS / (n - 2)) / np.sum((x - x_average) ** 2))
    b0_stdev = b1_stdev * np.sqrt(np.sum(x ** 2) / n)

    # compute the mean square error (variance) and standard error (root of var), R2 and R
    mserr = ResSS / (n - 2)
    sterr = np.sqrt(mserr)
    R2 = 1 - ResSS / TotSS
    R = np.sqrt(R2)

    ## calculate confidence interval
    alpha = 1. - (confidence_level * 1. / 100.)
    # degrees of freedom (2 lost by estimates of slope and intercept)
    DF = n - 2
    # critical value (from t table)
    cv = 2.01
    # Margin of error = Critical value x Standard error of the statistic 
    moe_b1 = cv * b1_stdev
    moe_b0 = cv * b0_stdev
    lower_b1 = b1 - moe_b1
    upper_b1 = b1 + moe_b1
    lower_b0 = b0 - moe_b0
    upper_b0 = b0 + moe_b0

    if report:
        print 'Report of linear regression:'
        print np.polyfit(x, y, 1)
        print 'Slope = %.6f +/- %.4f, Intercept = %.4f +/- %.4f' % (b1, moe_b1, b0, moe_b0)
        print '%g percent confidence interval for slope: %.4f to %.4f' % (confidence_level, lower_b1, upper_b1)
        print '%g percent confidence interval for intercept: %.4f to %.4f' % (confidence_level, lower_b0, upper_b0)

    # compute confidence lines for plot
    lower = upper_b1 * x + lower_b0
    upper = lower_b1 * x + upper_b0

    if show_plots:
        import matplotlib.pylab as plt

        plt.figure(1000)
        plt.title('My Linear Regression')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.plot(x, y, 'bo', label='data')
        plt.plot(x, y_hat, 'r.-', label='linear fit')
        plt.plot(x, lower, 'c-')
        plt.plot(x, upper, 'c-')
        plt.legend(loc='best', numpltoints=3)
        # are the residuals normally distributed?
        plt.figure(1001)
        plt.title('Residuals of fit')
        plt.xlabel('x')
        plt.ylabel('Residuals')
        plt.grid()
        plt.plot(x, y - y_hat, 'mo')
        plt.show()

    return b1, moe_b1, b0, lower_b1, upper_b1, lower_b0, upper_b0, R2, R


def FitExponent(xdata, ydata, initials):
    """
    Fits a negative exponential to x and y data with an initial guess given by initials.
    The form of the exponential is:

      p[0] + [1]*exp(-x*p[2])

    :param xdata: x vector
    :param ydata: y vector
    :param initials: initial parameter values
    """
    fitfunc = lambda p, x: p[0] + p[1] * scipy.exp(-x * p[2])
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    # fit
    p1, success = scipy.optimize.leastsq(errfunc, initials, args=(xdata, ydata))

    # compute the best fit function from the best fit parameters
    corrfit = fitfunc(p1, xdata)
    return corrfit, p1


def doubleExponent(x, p, shift):
    """
    :param x: data
    :param p: parameters
    :param shift: shift

    :return: double exponential with a given parameters and shift
    """
    return shift + p[0] + p[1] * scipy.exp(-x / p[2]) + p[3] * scipy.exp(-x / p[4])


def singleExponent(x, p, shift):
    """
    :param x: data
    :param p: parameters
    :param shift: shift

    :return: a single exponential with a given parameters and shift
    """
    return shift + p[0] + p[1] * scipy.exp(-x * p[2])


def FitDoubleExponent(xcorr, ycorr, initials):
    """
    Fits a negative double exponential to data.

    :return: best fit, fit parameters
    """
    fitfunc = lambda p, x: p[0] + p[1] * scipy.exp(-x / p[2]) + p[3] * scipy.exp(-x / p[4])
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    # fit
    p1, success = scipy.optimize.leastsq(errfunc, initials, args=(xcorr, ycorr))

    print 'Double exponent fit:'
    print 'p[0] + p[1]*scipy.exp(-x/p[2]) + p[3]*scipy.exp(-x/p[4])'
    print p1
    if success != 1: print success
    print

    # compute the best fit function from the best fit parameters
    corrfit = fitfunc(p1, xcorr)
    return corrfit, p1


def FindZeroDoubleExp(p, x0, yshift=0.0):
    """
    Finds the zero point of a negative double exponential function.

    :return: roots
    """
    roots = scipy.optimize.fsolve(doubleExponent, x0, args=(p, yshift))
    return roots


def FindZeroSingleExp(p, x0, yshift=0.0):
    """
    Finds the zero point of a negative exponential function.

    :return: roots
    """
    roots = scipy.optimize.fsolve(singleExponent, x0, args=(p, yshift))
    return roots


def PolyFit(x, y, order=1):
    """
    Fits a polynomial to the data.

    :return: fitted y values, error of the fit
    """
    (ar, br) = np.polyfit(y, x, order)
    yr = np.polyval([ar, br], y)
    err = np.sqrt(sum((yr - y) ** 2.) / len(yr))
    return yr, err
