#!/usr/bin/env python3

"""Implementation of the Kalman filter.
"""


from __future__ import annotations

import numpy as np
import numpy.typing as npt


__all__ = ['Kalman']


class Kalman:
    """Implementation of a Kalman filter.

    The implementaion of the code is based on Ref [1]_. The
    state transition matrix `F`, input transition matrix `F`
    and obervation matrix `H` are used to represent the system
    in state space, and should be correctly defined by the user.
    These matrixes can be defined when creating the Kalman instance
    or set by direct access to the property setter. That is:
        kalman_filter = Kalman(F, G, H)
    or
        kalman_filter = Kalman()
        kalman_filter.F = F
        kalman_filter.G = G
        kalman_filter.H = H
    Note that it is not necessary to set G if the system does
    not have input.

    The state vector and its uncertainty matrix can be accessed by
    attribute `x` and `P`. The initial values of these two variables
    should be manually defined after instantiation:
        kalman_filter.x = x
        kalman_filter.P = P

    After that, the filter is fully prepared for making predictions
    for the next timestep, or updating its state by new measurement:
        kalman_filter.predict(u, Q)
        kalman_filter.update(z, R)
    Users can refer to the docstring of these two member funcitons
    for the usage.

    Parameters
    ----------
    F: np.ndarray, optional
        The state transition matrix.
    G: np.ndarray, optional
        The input transition matrix.
    H: np.ndarray, optional
        The observation matrix.

    Notes
    -----
    It is the user's responsibility to make sure F, G, H and initial
    x and P are correctly set.

    References
    ----------
    .. [1] kalmanfilter.net
    """

    def __init__(self,
                 F: npt.ArrayLike | None = None,
                 G: npt.ArrayLike | None = None,
                 H: npt.ArrayLike | None = None):

        self._F = None          # state transition matrix
        self._G = None          # input transition matrix
        self._H = None          # observation matrix
        self._x = None          # state vector
        self._P = None          # covariance matrix of state vector

        self._K = None          # cache for Kalman gain
        self._Q = None          # cache for process noise uncertainty
        self._R = None          # cache for measurement uncertainty

        if F is not None:
            self.F = F
        if G is not None:
            self.G = G
        if H is not None:
            self.H = H

    @property
    def F(self) -> npt.NDArray | None:
        return self._F

    @F.setter
    def F(self, F: npt.ArrayLike):
        self._F = np.asarray(F)

    @property
    def G(self) -> npt.NDArray | None:
        return self._G

    @G.setter
    def G(self, G: npt.ArrayLike):
        self._G = np.asarray(G)

    @property
    def H(self) -> npt.NDArray | None:
        return self._H

    @H.setter
    def H(self, F: npt.ArrayLike):
        self._H = np.asarray(F)

    @property
    def x(self) -> npt.NDArray | None:
        return self._x

    @x.setter
    def x(self, x: npt.ArrayLike):
        if self._x is not None:
            raise AttributeError('x has already been set')
        self._x = np.asarray(x)

    @property
    def P(self) -> npt.NDArray | None:
        return self._P

    @P.setter
    def P(self, P: npt.ArrayLike):
        if self._P is not None:
            raise AttributeError('P has already been set')
        self._P = np.asarray(P)

    @property
    def K(self) -> npt.NDArray | None:
        """Read-only access to Kalman gain."""
        return self._K

    def predict(self, u: npt.ArrayLike | None = None,
                Q: npt.ArrayLike | None = None) -> Kalman:
        """Predict the system state at the next step.

        Both the state vector and its uncertainty at the
        next timestep are predicted by given input
        vector `u` and its uncertainty matrix `Q`.

        Note that its the user's responsibility to make
        sure the dimensions of the input arguments are
        correct.

        Parameters
        ----------
        u : np.ndarray, optional
            The input variable (control vector). If not given,
            the system input is assumed to be zero, thus the term
            concerning the input transition matrix `G` is ignored.
            (defaults to None)
        Q : np.ndarray, optional
            The process noise uncertainty of the timestep. If not
            given, the last setted value for `Q` will be used.
            (defaults to None)

        Returns
        -------
        self
            The instance itself.

        Raises
        ------
        ValueError
            If the system matrixes or initial values are not set.
        """
        try:
            return self._predict_helper(u, Q)
        # Additional error handling for better traceback.
        except TypeError:
            if self.F is None:
                raise AttributeError("F not set") from None
            if self.G is None and u is not None:
                raise AttributeError("G not set") from None
            if self.x is None:
                raise AttributeError("initial x not set") from None
            if self.P is None:
                raise AttributeError("initial P not set") from None
            raise

    def _predict_helper(self, u: npt.ArrayLike | None,
                        Q: npt.ArrayLike | None) -> Kalman:
        """The workhorse of `predict`."""
        # Calculate predicted state vector `x`.
        if u is None:
            x = self.F @ self.x
        else:
            x = self.F @ self.x + self.G @ np.asarray(u)
        # Calculate the uncertainty `P` of state vector.
        if Q is not None:
            self._Q = np.asarray(Q)
        P = self.F @ self._P @ self.F.T + self._Q
        # Update self.x and self.P.
        self._x = x
        self._P = P
        return self

    def update(self, z: npt.NDArray,
               R: npt.NDArray | None = None) -> Kalman:
        """Update the system state by measurement.

        Both the state vector and its uncertainty at the
        next timestep are updated by given measurement
        vector `z` and its uncertainty matrix `R`.

        Note that its the user's responsibility to make
        sure the dimensions of the input arguments are
        correct.

        Parameters
        ----------
        z : np.ndarray
            The measured output of the system.
        R : np.ndarray, optional
            The measurement uncertainty. If not given, the last
            setted value for `R` will be used.
            (defaults to None)

        Returns
        -------
        self
            The instance itself.

        Raises
        ------
        ValueError
            If the system matrixes or initial values are not set.
        """
        try:
            return self._update_helper(z, R)
        # Additional error handling for better traceback.
        except TypeError:
            if self.H is None:
                raise AttributeError("H not set") from None
            if self.x is None:
                raise AttributeError("initial x not set") from None
            if self.P is None:
                raise AttributeError("initial P not set") from None
            raise

    def _update_helper(self, z: npt.ArrayLike,
                       R: npt.ArrayLike | None) -> Kalman:
        """The workhorse of `update`."""
        if R is not None:
            self._R = np.asarray(R)
        # Calculate Kalman gain `K`.
        K = (self.P @ self.H.T @
             np.linalg.inv(self.H @ self.P @ self.H.T + self._R))
        # Calculate the updated state vector `x`.
        x = self.x + K @ (np.asarray(z) - self.H @ self.x)
        # Calculate the updated uncertainty `P` for the state vector.
        T = np.eye(K.shape[0]) - K @ self.H
        P = T @ self.P @ T.T + K @ self._R @ K.T
        # Update self.K, self.x and self.P.
        self._K = K
        self._x = x
        self._P = P
        return self
