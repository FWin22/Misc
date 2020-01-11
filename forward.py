import numpy as np
import math
import jutil.diff as jdiff
from jutil import fft
from scipy.sparse import identity


def create_empty_aber_dict():
    """
    Resets all aberrations to zero.
    """
    aberrations = {'A0': 0. + 1j * 0., 'A1': 0. + 1j * 0., 'C1': 0. + 1j * 0.,
                   'A2': 0. + 1j * 0., 'B2': 0. + 1j * 0., 'A3': 0. + 1j * 0.,
                   'C3': 0. + 1j * 0., 'S3': 0. + 1j * 0., 'A4': 0. + 1j * 0.,
                   'B4': 0. + 1j * 0., 'D4': 0. + 1j * 0., 'C5': 0. + 1j * 0.}

    return aberrations


def chi(wavelength, w, aberrations):
    """
    Returns aberration function chi(w). All aberrations are in nm.

    Aberrations
    ----------
    A0 : complex, optional
        Image Shift
    C1 : float, optional
        Defocus
    A1 : complex, optional
        2-fold-astigmatism
    A2 : complex, optional
        3-fold-astigmatism
    B2 : complex, optional
        Axial coma
    C3 : float, optional
        Spherical aberration
    A3 : complex, optional
        4-fold-astigmatism
    S3 : complex, optional
        Star aberration
    A4 : complex, optional
        5-fold-astigmatism
    B4 : complex, optional
        Axial coma
    D4 : complex, optional
        Three lobe aberration
    C5 : float, optional
        Spherical aberration
    """

    w_c = np.conjugate(w)
    chi_i = {}
    chi_i['A0'] = aberrations['A0'] * w_c
    chi_i['C1'] = aberrations['C1'] * w * w_c / 2
    chi_i['A1'] = (aberrations['A1'] * w_c ** 2) / 2
    chi_i['B2'] = aberrations['B2'] * w * w_c ** 2
    chi_i['A2'] = (aberrations['A2'] * w_c ** 3) / 3
    chi_i['A3'] = (aberrations['A3'] * w_c ** 4) / 4
    chi_i['C3'] = (aberrations['C3'] * w * w * w_c * w_c) / 4
    chi_i['S3'] = aberrations['S3'] * w * w_c ** 3
    chi_i['A4'] = (aberrations['A4'] * w_c ** 5) / 5
    chi_i['B4'] = aberrations['B4'] * w ** 2 * w_c ** 3
    chi_i['D4'] = aberrations['D4'] * w * w_c ** 4
    chi_i['C5'] = (aberrations['C5'] * w ** 3 * w_c ** 3) / 6

    chi_sum = 0 * w
    for key in chi_i.keys():
        chi_sum += chi_i[key]

    return (2 * math.pi / wavelength) * np.real(chi_sum)


def diff_chi_ab(wavelength, w, aber_list, gmax):
    w_c = np.conjugate(w)
    chi_i = {}
    chi_i['A0'] = w_c
    chi_i['C1'] = (w * w_c) / 2
    chi_i['A1'] = (w_c ** 2) / 2
    chi_i['B2'] = w * w_c ** 2
    chi_i['A2'] = (w_c ** 3) / 3
    chi_i['A3'] = (w_c ** 4) / 4
    chi_i['C3'] = (w * w * w_c * w_c) / 4
    chi_i['S3'] = w * w_c ** 3
    chi_i['A4'] = (w_c ** 5) / 5
    chi_i['B4'] = w ** 2 * w_c ** 3
    chi_i['D4'] = w * w_c ** 4
    chi_i['C5'] = (w ** 3 * w_c ** 3) / 6

    a = []
    for i, ab in enumerate(aber_list):
        if gmax:
            dx = (2 * math.pi / wavelength * np.real(chi_i[ab]) * pi_4th(ab, wavelength, gmax))
            dy = (2 * math.pi / wavelength * np.real(1j*chi_i[ab]) * pi_4th(ab, wavelength, gmax))
        else:
            dx = (2 * math.pi / wavelength * np.real(chi_i[ab]) )
            dy = (2 * math.pi / wavelength * np.real(1j * chi_i[ab]))
        a.append(dx)
        a.append(dy)

    a = np.asarray(a)

    return a.T


def diff_chi_x_ab(wavelength, w, aber_list, gmax=None):
    x = np.real(w).copy()
    y = np.imag(w).copy()
    w_c = np.conjugate(w)

    chi_i = {}
    chi_i['A0'] = np.zeros(w.shape, dtype='complex128') #np.ones(w.shape, dtype='complex128')
    chi_i['A1'] = w_c
    chi_i['A2'] = w_c ** 2
    chi_i['A3'] = w_c ** 3
    chi_i['A4'] = w_c ** 4
    chi_i['C1'] = x
    chi_i['C3'] = x * w * w_c
    chi_i['C5'] = x * w ** 2 * w_c ** 2
    chi_i['B2'] = (3 * x ** 2 + y ** 2 - 2j * x * y)
    chi_i['B4'] = w_c ** 2 * (5 * x ** 2 + 6j * x * y - y ** 2)
    chi_i['S3'] = 2 * w_c ** 2 * (2 * x + 1j * y)
    chi_i['D4'] = w_c ** 3 * (5 * x + 3j * y)

    a = []
    for i, ab in enumerate(aber_list):
        dx = (2 * np.pi / wavelength) * np.real(chi_i[ab]).ravel()
        dy = (-2 * np.pi / wavelength) * np.imag(chi_i[ab]).ravel()
        a.append(dx)
        a.append(dy)
    a = np.asarray(a)

    return a


def diff_chi_y_ab(wavelength, w, aber_list, gmax=None):
    x = np.real(w).copy()
    y = np.imag(w).copy()
    w_c = np.conjugate(w)

    chi_i = {}
    chi_i['A0'] = np.zeros(w.shape, dtype='complex128')# (-1j) * np.ones(w.shape,
    # dtype='complex128')
    chi_i['A1'] = (-1j) * w_c
    chi_i['A2'] = (-1j) * w_c ** 2
    chi_i['A3'] = (-1j) * w_c ** 3
    chi_i['A4'] = (-1j) * w_c ** 4
    chi_i['C1'] = y
    chi_i['C3'] = y * w * w_c
    chi_i['C5'] = y * w ** 2 * w_c ** 2
    chi_i['B2'] = (-1j) * (3 * y ** 2 + x ** 2 + 2j * x * y)
    chi_i['B4'] = w_c ** 2 * (5j * y ** 2 - 1j * x ** 2 + 6 * x * y)
    chi_i['S3'] = 2 * w_c ** 2 * (2 * y - 1j * x)
    chi_i['D4'] = w_c ** 3 * (5 * y - 3j * x)

    a = []
    for i, ab in enumerate(aber_list):
        dx = (2 * np.pi / wavelength) * np.real(chi_i[ab]).ravel()
        dy = (-2 * np.pi / wavelength) * np.imag(chi_i[ab]).ravel()
        a.append(dx)
        a.append(dy)
    a = np.asarray(a)

    return a


def diff_chi_x(wavelength, w, aberrations):
    x = np.real(w)
    y = np.imag(w)
    w_c = np.conjugate(w)

    chi_i = {}
    chi_i['A0'] = np.zeros(w.shape, dtype='complex128') #aberrations['A0'] * np.ones(w.shape)
    chi_i['A1'] = aberrations['A1'] * w_c
    chi_i['A2'] = aberrations['A2'] * w_c ** 2
    chi_i['A3'] = aberrations['A3'] * w_c ** 3
    chi_i['A4'] = aberrations['A4'] * w_c ** 4
    chi_i['C1'] = aberrations['C1'] * x
    chi_i['C3'] = aberrations['C3'] * x * w * w_c
    chi_i['C5'] = aberrations['C5'] * x * w ** 2 * w_c ** 2
    chi_i['B2'] = aberrations['B2'] * (3 * x ** 2 + y ** 2 - 2j * x * y)
    chi_i['B4'] = aberrations['B4'] * w_c ** 2 * (5 * x ** 2 + 6j * x * y - y ** 2)
    chi_i['S3'] = aberrations['S3'] * 2 * w_c ** 2 * (2 * x + 1j * y)
    chi_i['D4'] = aberrations['D4'] * w_c ** 3 * (5 * x + 3j * y)

    chi_sum = 0 * w
    for key in chi_i.keys():
        chi_sum += chi_i[key]

    return (2 * math.pi / wavelength) * np.real(chi_sum)


def diff_chi_y(wavelength, w, aberrations):
    x = np.real(w)
    y = np.imag(w)
    w_c = np.conjugate(w)

    chi_i = {}
    chi_i['A0'] = np.zeros(w.shape, dtype='complex128')# aberrations['A0'] * (-1j) * np.ones(
    # w.shape)
    chi_i['A1'] = aberrations['A1'] * (-1j) * w_c
    chi_i['A2'] = aberrations['A2'] * (-1j) * w_c ** 2
    chi_i['A3'] = aberrations['A3'] * (-1j) * w_c ** 3
    chi_i['A4'] = aberrations['A4'] * (-1j) * w_c ** 4
    chi_i['C1'] = aberrations['C1'] * y
    chi_i['C3'] = aberrations['C3'] * y * w * w_c
    chi_i['C5'] = aberrations['C5'] * y * w ** 2 * w_c ** 2
    chi_i['B2'] = aberrations['B2'] * (-1j) * (3 * y ** 2 + x ** 2 + 2j * x * y)
    chi_i['B4'] = aberrations['B4'] * w_c ** 2 * (5j * y ** 2 - 1j * x ** 2 + 6 * x * y)
    chi_i['S3'] = aberrations['S3'] * 2 * w_c ** 2 * (2 * y - 1j * x)
    chi_i['D4'] = aberrations['D4'] * w_c ** 3 * (5 * y - 3j * x)

    chi_sum = 0 * w
    for key in chi_i.keys():
        chi_sum += chi_i[key]

    return (2 * math.pi / wavelength) * np.real(chi_sum)


def pi_4th(aberration, wavelength, gmax):
    """
    Determines the pi/4 limit of the given aberration, electron wavelength and gmax.
    Parameters
    ----------
    aberration : str
        Aberration: 'A0', 'A1', 'C1', ...
    wavelength : float
        in nm.
    gmax : float
        in nm^-1.
    """

    lim = 0
    if aberration == 'A0':
        lim = 1 / (8 * gmax)
    elif aberration == 'C1':
        lim = 1 / (4 * wavelength * (gmax ** 2))
    elif aberration == 'A1':
        lim = 1 / (4 * wavelength * (gmax ** 2))
    elif aberration == 'A2':
        lim = 3 / (8 * (wavelength ** 2) * (gmax ** 3))
    elif aberration == 'B2':
        lim = 1 / (8 * (wavelength ** 2) * (gmax ** 3))
    elif aberration == 'C3' or aberration == 'A3':
        lim = 1 / (2 * (wavelength ** 3) * (gmax ** 4))
    elif aberration == 'S3':
        lim = 1 / (8 * (wavelength ** 3) * (gmax ** 4))
    elif aberration == 'A4':
        lim = 5 / (8 * (wavelength ** 4) * (gmax ** 5))
    elif aberration == 'B4' or aberration == 'D4':
        lim = 1 / (8 * (wavelength ** 4) * (gmax ** 5))
    elif aberration == 'C5' or aberration == 'A5':
        lim = 3 / (4 * (wavelength ** 5) * (gmax ** 6))

    return lim


class ForwardModel(object):

    def __init__(self, exp, sim, w_2d, wavelength, aber_list, offset=None, gmax=None,
                 phase_norm=False):
        self.exp = exp
        self.sim = sim
        if phase_norm:
            phase_diff = np.angle(np.mean(self.exp)) - np.angle(np.mean(self.sim))
            self.exp = exp / np.exp(1j * phase_diff)

        self.shape = self.exp.shape
        self.fft_exp = fft.fftn(self.exp) / np.prod(self.shape)
        self.fft_sim = fft.fftn(self.sim) / np.prod(self.shape)
        self.wavelength = wavelength
        self.aber_list = aber_list

        self.y = np.concatenate((self.fft_sim.real.ravel(), self.fft_sim.imag.ravel()))
        #self.y = np.concatenate((self.sim.real.ravel(), self.sim.imag.ravel()))
        self.n = 2 * len(self.aber_list)
        self.m = len(self.y)
        self.Se_inv = identity(self.m)

        self.w_2d = w_2d
        self.gmax = gmax

        if offset is None:
            self.offset = create_empty_aber_dict()
        else:
            self.offset = offset

    def aber_dict(self, x):
        aber_dict = create_empty_aber_dict()

        for i, ab in enumerate(self.aber_list):
            z = x[i * 2] + 1j * x[i * 2 + 1]
            if self.gmax:
                aber_dict[ab] = z * pi_4th(ab, self.wavelength, self.gmax)
            else:
                aber_dict[ab] = z

        for ab in self.offset.keys():
            aber_dict[ab] += self.offset[ab]

        return aber_dict

    def __call__(self, x):
        cq = chi(self.wavelength, self.w_2d, self.aber_dict(x))
        exp_corrected = self.fft_exp * np.exp(1j * cq)
        # exp_corrected = fft.ifftn(self.fft_exp * np.exp(1j * cq)) * np.prod(self.shape)
        return np.concatenate((exp_corrected.real.ravel(), exp_corrected.imag.ravel()))

    def chi(self, x):
        return chi(self.wavelength, self.w_2d, self.aber_dict(x))

    def apply_aberrations(self, x):
        cq = self.chi(x)
        return fft.ifftn(self.fft_exp * np.exp(1j * cq)) * np.prod(self.shape)

    def jac(self, x):

        cq = chi(self.wavelength, self.w_2d, self.aber_dict(x))
        da = diff_chi_ab(self.wavelength, self.w_2d, self.aber_list, self.gmax)

        h1 = (1j * self.fft_exp * np.exp(1j * cq)) * da.T

        f = []
        for g in h1:
            f.append(np.concatenate((g.real.ravel(), g.imag.ravel())))
        f = np.asarray(f)
        return f.T

    def jac_dot(self, x, vec):
        return self.jac(x).dot(vec)

    def jac_T_dot(self, x, vec):
        jac_T = self.jac(x).T
        return jac_T.dot(vec)

    def hess(self, x):
        cq = chi(self.wavelength, self.w_2d, self.aber_dict(x))

        h = []
        for ab in self.aber_list:
            da = diff_chi_ab(self.wavelength, self.w_2d, [ab], self.gmax)

            da_x = np.concatenate((da.T[0].real.ravel(), da.T[0].imag.ravel()))
            da_y = np.concatenate((da.T[1].real.ravel(), da.T[1].imag.ravel()))
            h.append(da_x)
            h.append(da_y)
        h = np.asarray(h)

        f = []
        for ab in self.aber_list:
            da = diff_chi_ab(self.wavelength, self.w_2d, [ab], self.gmax)
            da_xx = - da.T[0] * (1j * self.fft_exp * np.exp(1j * cq))
            da_yy = - da.T[1] * (1j * self.fft_exp * np.exp(1j * cq))
            da_x = np.concatenate((da_xx.real.ravel(), da_xx.imag.ravel()))
            da_y = np.concatenate((da_yy.real.ravel(), da_yy.imag.ravel()))

            f.append((da_x * h).T)
            f.append((da_y * h).T)
        f = np.asarray(f)

        return f

    def hess_dot(self, x, vec):
        return self.hess(x).dot(vec).T

    def estimate_std(self, x):
        f0 = self(x)
        delta_y = self.y - f0

        chi_sq = (delta_y ** 2).sum() / (self.m - self.n)
        # chi_sq = np.linalg.norm(delta_y) ** 2 / (self.m + 1 - self.n)

        A = self.jac(x)

        ATA = A.T.dot(A)

        x_var = np.diag(np.linalg.pinv(ATA) * chi_sq)
        x_std = np.sqrt(np.abs(x_var))

        aber_dict = create_empty_aber_dict()
        for i, ab in enumerate(self.aber_list):
            z = x_std[i * 2] + 1j * x_std[i * 2 + 1]
            if self.gmax:
                aber_dict[ab] = z * pi_4th(ab, self.wavelength, self.gmax)
            else:
                aber_dict[ab] = z
        return aber_dict


