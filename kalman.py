#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Kalman:
    """Implementation of a Kalman filter.

    Parameters
    ----------
    F: np.ndarray, optional
        The state transition matrix.
    G: np.ndarray, optional
        The input transition matrix.
    H: np.ndarray, optional
        The observation matrix.
        
    
    """

    def __init__(self,
                 F: npt.NDArray | None = None,
                 G: npt.NDArray | None = None,
                 H: npt.NDArray | None = None):

        self._F = None          # state transition matrix
        self._G = None          # input transition matrix
        self._H = None          # observation matrix
        self._x = None          # state vector
        self._P = None          # covariance matrix of state vector
        self._Q = None          # cache for process noise uncertainty

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
        # Additional error handling for better error messages.
        except TypeError:
            if self.F is None:
                raise AttributeError("F not set") from None
            if self.H is None:
                raise AttributeError("H not set") from None
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
        Q = self._Q
        P = self.F @ self._P @ self.F.T + Q
        # Update self.x and self.P.
        self._x = x
        self._P = P
        return self
