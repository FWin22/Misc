import numpy as np
from Stuff import usefulfunctions as uf
import scipy.optimize as opt
import matplotlib.pyplot as plt


def func_linear(x, a, b):
    return a * x + b


def gauss_1d(x, mu, sigma):
    return 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(- (x - mu)**2 / (2*sigma**2))


def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    # g = offset + amplitude * np.exp(-1 * (((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2)
    #                                       / (2 * sigma_y ** 2)))
    return g.ravel()


def sphere(xdata_tuple, radius, x0, y0):
    (x, y) = xdata_tuple

    f = radius ** 2 - (x - x0) ** 2 - (y - y0) ** 2
    s = np.sqrt(np.where(f > 0, f, 0))

    return s / np.max(s)


def gauss_2D(xdata_tuple, amplitude, x0, y0, sigma_x, sigma_y, offset):
    (x, y) = xdata_tuple
    g = offset + amplitude * np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))
    return g.ravel()


def gauss_2D_der_x(xdata_tuple, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xdata_tuple
    f = gauss_2D(xdata_tuple, amplitude, x0, y0, sigma_x, sigma_y, offset)
    f = f.reshape(xdata_tuple[0].shape)
    return (x0 - x) / (sigma_x ** 2) * f


def gauss_2D_der_y(xdata_tuple, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xdata_tuple
    f = gauss_2D(xdata_tuple, amplitude, x0, y0, sigma_x, sigma_y, offset)
    f = f.reshape(xdata_tuple[0].shape)
    return (y0 - y) / (sigma_y ** 2) * f


def gauss_peak_finder_2d(data, peaks, width=None, full_params=False, verbose=False, maxit=10000,
                         peak_shape='2D_Gauss', cov=False):
    """Returns a list of fitted peak parameters:

    Height, y, x, width_y, width_x, theta, offset
    Parameters
    ----------
    data : ndarray
        The 2d data on which the peaks will be fitted.
    peaks : tuples
        A list of approximate candidates for the peaks. y, x
    width : int
        The width (radius) around each peak, where the fit will be performed."""

    if width is None:
        width = int(np.ceil(uf.mean_min_distance(peaks) / (2.*np.sqrt(2))))

    fitted_peaks = []
    fitted_covs = []

    for peak in peaks:
        y_min = int(peak[0] - width)
        y_max = int(peak[0] + width)
        x_min = int(peak[1] - width)
        x_max = int(peak[1] + width)
        if y_min < 0:
            y_min = 0
        if y_max > data.shape[0]:
            y_max = int(data.shape[0])
        if x_min < 0:
            x_min = 0
        if x_max > data.shape[1]:
            x_max = int(data.shape[1])

        cut = data[y_min:y_max, x_min:x_max]

        x = np.arange(0, cut.shape[1], 1)
        y = np.arange(0, cut.shape[0], 1)

        x, y = np.meshgrid(x, y)

        height = data[int(peak[0]), int(peak[1])]

        x0 = int(width)
        y0 = int(width)

        if height > np.mean(data[int(peak[0])-1:int(peak[0])+1, int(peak[1])-1:int(peak[1])+1]):
            offset = np.min(cut)
        else:
            offset = np.max(cut)

        if verbose:
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(cut)
            axs[1].plot(cut[:, y0])
            axs[2].plot(cut[x0, :])
            plt.show()
            print('height = {}'.format(height))
            print('X0 = {}, Y0 = {}'.format(x0, y0))
            print('Offset = {}'.format(offset))

        if peak_shape == '2D_Gauss':
            initial_guess = (height, x0, y0, x0 / 2., y0 / 2., offset)
            popt, pcov = opt.curve_fit(gauss_2D, (x, y), cut.ravel(), p0=initial_guess,
                                       max_nfev=maxit,
                                       bounds=([-np.inf, 0, 0, 0, 0, -np.inf],
                                               [np.inf, 2 * width, 2 * width, 2 * width, 2 * width,
                                                np.inf]))
            if full_params:
                fitted_peaks.append((popt[0], popt[2] + peak[0] - width, popt[1] + peak[1] - width,
                                     popt[4], popt[3], popt[5]))
                fitted_covs.append(pcov)
            else:
                fitted_peaks.append([popt[2] + peak[0] - width, popt[1] + peak[1] - width])
                fitted_covs.append(pcov)
        elif peak_shape == '2D_Gauss_asym':
            initial_guess = (height, x0, y0, x0 / 2., y0 / 2., 0, offset)
            popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), cut.ravel(), p0=initial_guess,
                                   max_nfev=maxit,
                                   bounds=([-np.inf, 0, 0, 0, 0, 0, -np.inf],
                                           [np.inf, 2*width, 2*width, 2*width, 2*width, 2*np.pi,
                                            np.inf]))

            if full_params:
                fitted_peaks.append((popt[0], popt[2] + peak[0] - width, popt[1] + peak[1] - width,
                                     popt[4], popt[3], popt[5], popt[6]))
                fitted_covs.append(pcov)
            else:
                fitted_peaks.append([popt[2] + peak[0] - width, popt[1] + peak[1] - width])
                fitted_covs.append(pcov)

    fitted_peaks = np.asarray(fitted_peaks)
    fitted_covs = np.array(fitted_covs)

    if cov:
        return fitted_peaks, fitted_covs
    else:
        return fitted_peaks


def polynomial(poly_order_x, poly_order_y):
    def _polynom(xdata_tuple, *p):
        y, x = xdata_tuple
        p = np.array(p)
        f = []
        for dx in range(poly_order_x + 1):
            for dy in range(poly_order_y + 1):
                if dx + dy <= np.max((poly_order_x, poly_order_y)):
                    f.append((x ** dx) * (y ** dy))
        f = np.asarray(f)
        return f.T.dot(p).T

    return _polynom


def polynomial_sigma(poly_order_x, poly_order_y):
    def _polynom(xdata_tuple, *p):
        y, x = xdata_tuple
        p = np.array(p)
        f, params = [], []
        for dx in range(poly_order_x + 1):
            for dy in range(poly_order_y + 1):
                if dx + dy <= np.max((poly_order_x, poly_order_y)):
                    f.append((x ** dx) * (y ** dy))
                    params.append('{}x{}y'.format(dx, dy))
        f = np.asarray(f) ** 2
        return np.sqrt(f.T.dot(p ** 2).T)

    return _polynom


def _polynomial_params(poly_order_x, poly_order_y):
    params = []
    for dx in range(poly_order_x + 1):
        for dy in range(poly_order_y + 1):
            if dx + dy <= np.max((poly_order_x, poly_order_y)):
                params.append('{}x{}y'.format(dx, dy))
    return params