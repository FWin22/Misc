
#
"""This module provides the :class:`~.Costfunction` class which represents a strategy to calculate
the so called `cost` of a threedimensional magnetization distribution."""

import logging

import numpy as np

from removeaberrations import regularisator as rg
from _collections import deque
import jutil.diff as jdiff

__all__ = ['Costfunction']


class Costfunction(object):
    """Class for calculating the cost of a 3D magnetic distributions in relation to 2D phase maps.

    Represents a strategy for the calculation of the `cost` of a 3D magnetic distribution in
    relation to two-dimensional phase maps. The `cost` is a measure for the difference of the
    simulated phase maps from the magnetic distributions to the given set of phase maps and from
    a priori knowledge represented by a :class:`~.Regularisator` object. Furthermore this class
    provides convenient methods for the calculation of the derivative :func:`~.jac` or the product
    with the Hessian matrix :func:`~.hess_dot` of the costfunction, which can be used by
    optimizers. All required data should be given in a :class:`~DataSet` object.

    Attributes
    ----------
    regularisator : :class:`~.Regularisator`
        Regularisator class that's responsible for the regularisation term.
    y : :class:`~numpy.ndarray` (N=1)
        Vector which lists all pixel values of all phase maps one after another.
    fwd_model : :class:`~.ForwardModel`
        The Forward model instance which should be used for the simulation of the phase maps which
        will be compared to `y`.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `NxN` with N
        being the length of the targetvector y (vectorized phase map information).
    m: int
        Size of the image space.
    n: int
        Size of the input space.

    """

    _log = logging.getLogger(__name__ + '.Costfunction')

    def __init__(self, fwd_model, regularisator=None, lam=0):
        self._log.debug('Calling __init__')
        self.fwd_model = fwd_model
        if regularisator is None:
            self.regularisator = rg.NoneRegularisator()
        else:
            self.regularisator = regularisator
        # Extract information from fwd_model:
        self.y = self.fwd_model.y
        self.n = self.fwd_model.n
        self.m = self.fwd_model.m
        self.lam = lam
        self.jacs = deque(maxlen=20)
        self.jacs_rg = deque(maxlen=20)
        self.jac_keys = deque(maxlen=20)
        self.Se_inv = self.fwd_model.Se_inv
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(fwd_model=%r, regularisator=%r)' % \
               (self.__class__, self.fwd_model, self.regularisator)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'Costfunction(fwd_model=%s, fwd_model=%s, regularisator=%s)' % \
               (self.fwd_model, self.fwd_model, self.regularisator)

    def __call__(self, x):
        # delta_y = self.fwd_model(x) - self.y
        # self.chisq_m = delta_y.dot(self.Se_inv.dot(delta_y))
        # self.chisq_a = self.regularisator(x)
        # # print('s: {:>8.1f}  - reg: {:>8.1f}'.format(self.chisq_m, self.chisq_a))
        # self.chisq = self.chisq_m + self.chisq_a
        # return self.chisq
        #w = np.concatenate((np.abs(self.fwd_model.w_2d).ravel(), np.abs(
        #    self.fwd_model.w_2d).ravel())) / np.max(np.abs(self.fwd_model.w_2d))

        if x.ndim == 1:
            #if self.lam:
            #    delta_y = np.sqrt(w * self.lam) * (self.fwd_model(x[0]) - self.y)
            #else:
            #    delta_y = self.fwd_model(x[0]) - self.y
            delta_y = self.fwd_model(x) - self.y
            self.chisq_m = delta_y.dot(self.Se_inv.dot(delta_y))
            self.chisq_a = self.regularisator(x)
            self.chisq = self.chisq_m + self.chisq_a
            return self.chisq
        else:
            self.chisq = []
            for x0 in x:
                # if self.lam:
                #     delta_y = np.sqrt(w * self.lam) * (self.fwd_model(x0) - self.y)
                # else:
                #     delta_y = self.fwd_model(x0) - self.y
                delta_y = self.fwd_model(x0) - self.y
                self.chisq_m = delta_y.dot(self.Se_inv.dot(delta_y))
                self.chisq_a = self.regularisator(x0)
                self.chisq.append(self.chisq_m + self.chisq_a)
            return np.array(self.chisq)

    def init(self, x):
        """Initialise the costfunction by calculating the different cost terms.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution, for which the cost is calculated.

        Returns
        -------
        None

        """
        self._log.debug('Calling init')
        self(x)

    def jac(self, x):
        """Calculate the derivative of the costfunction for a given magnetization distribution.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution, for which the Jacobi vector is calculated.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Jacobi vector which represents the cost derivative of all voxels of the magnetization.

        """
        return (2 * self.fwd_model.jac_T_dot(x, self.Se_inv.dot(self.fwd_model(x) - self.y))
                + self.regularisator.jac(x))

    def hess_dot(self, x, vec):
        """Calculate the product of a `vector` with the Hessian matrix of the costfunction.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution at which the Hessian is calculated. The Hessian
            is constant in this case, thus `x` can be set to None (it is not used int the
            computation). It is implemented for the case that in the future nonlinear problems
            have to be solved.
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution which is multiplied by the Hessian.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the input `vector` with the Hessian matrix of the costfunction.

        """

        # Noch Falsch
        #return 2 * (self.fwd_model.jac_T_dot(x, self.Se_inv.dot(self.fwd_model.jac_dot(x, vec)))
        #            + self.fwd_model.hess_dot(x, self.Se_inv.dot(self.fwd_model(x))))

        a = self.fwd_model.hess_dot(x, vec).T.dot((self.fwd_model(x) - self.y))
        b = self.fwd_model.jac_T_dot(x, self.Se_inv.dot(self.fwd_model.jac_dot(x, vec)))
        return 2 * (a + b)
        #return jdiff.fd_hess_dot(self, x, vec) * 1000

    def hess_diag(self, _):
        """ Return the diagonal of the Hessian.

        Parameters
        ----------
        _ : undefined
            Unused input

        """
        return np.ones(self.n)
