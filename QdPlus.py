# /*
#  * @Author: Jingzhe Luo 
#  * @Date: 2023-11-09 23:01:01 
#  * @Last Modified by:   Jingzhe Luo 
#  * @Last Modified time: 2023-11-14 23:01:01
#  */

import numpy as np
import math as math
import scipy.optimize
import utils
from EuropeanPricing import EuropeanOption


class QdInitial:
    K: float

    def __init__(self, r: float, q: float, S: float, strike: float, vol: float, tau_vec: object) -> None:
        self.euro_solver = None
        self.r = r
        self.q = q
        self.S = S
        self.K = strike
        self.vol = vol
        self.tau_vec = tau_vec
        self.omega = 0.0
        self.h = 0.0
        self.euro_theta = 0.0
        self.euro = 0.0

    def init_price(self):
        boundaries = np.zeros(np.size(self.tau_vec))
        self.euro_solver = EuropeanOption(self.r, self.q, self.S, self.K, self.vol, "Put")
        for j in range(1, np.size(self.tau_vec)):
            tau = self.tau_vec[0, j]
            self.euro_solver.tau = tau
            boundaries[j] = self._find_root(tau)
        return boundaries[1:]

    def _find_root(self, _tau):
        # !!TODO: look into usage of scipy.optimize.root, need correction here
        ans = scipy.optimize.root(self._fun, x0=self.K, args=(_tau,))
        return ans.x

    def _fun(self, b, tau):
        self._compute_parts(b, tau)
        ans = self.left + self.right
        return ans

    def _compute_parts(self, b, tau):
        self.left = 1 - np.exp(-self.q * tau) * utils.phi_(-utils.d1_(tau, b / self.K, self.r, self.q, self.vol))

        self.euro_solver.S = b
        self.euro = self.euro_solver._europ_solver()
        self.euro_theta = self.euro_solver.euro_theta_put()

        self.omega = 2 * (self.r - self.q) / (self.vol ** 2)
        self.h = 1 - math.exp(-self.r * tau)
        self._c0_(b, tau)
        self.right = (self._lambda + self._c0) * (self.K - b - self.euro) / b

    def _lambda_(self):
        self._lambda = 0.5 * (1 - self.omega - math.sqrt((self.omega - 1) ** 2 + 8 * self.r / (self.h * self.vol ** 2)))

    def _c0_(self, b, tau):
        self._lambda_()
        lambda_d = 2 * self.r / (
                    math.sqrt((self.omega - 1) ** 2 + 8 * self.r / (self.h * self.vol ** 2)) * (self.vol * self.h) ** 2)
        base_a = 2 * self._lambda + self.omega - 1
        self._c0 = 1 / self.h
        self._c0 += lambda_d / base_a
        self._c0 -= math.exp(self.r * tau) * self.euro_theta / (self.r * (self.K - b - self.euro))
        self._c0 = -(1 - self.h) * 2 * self.r * self._c0 / (base_a * self.vol ** 2)
