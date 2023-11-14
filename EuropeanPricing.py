# /*
#  * @Author: Jingzhe Luo 
#  * @Date: 2023-11-09 23:01:30 
#  * @Last Modified by:   Jingzhe Luo 
#  * @Last Modified time: 2023-11-09 23:01:30 
#  */

import math as math
import scipy.stats as ss


class EuropeanOption():
    def __init__(self, r=None, q=None, S=None, K=None, vol=None, tau=None, option_type="Put") -> None:
        self.q = q
        self.vol = vol
        self.r = r
        self.K = K
        self.S = S
        self.tau = tau
        self.option_type = option_type

    def _europ_solver(self):
        if self.option_type == "Call":
            return self.__Euro_call__()
        else:
            return self.__Euro_put__()

    def __N__(self, d):
        """
        Find distribution function value for standard normal distribution 
        Args:
            d: given value
        Returns:
            return: P(N<=d)
        """
        return ss.norm.cdf(d)

    def __d1_d2__(self):
        """
        calculate d1,d2
        Returns:
            return: d1 and d2
        """

        d1 = (math.log(self.S / self.K) +
              (self.r - self.q + 0.5 * self.vol ** 2) *
              self.tau) / (self.vol * math.sqrt(self.tau))

        d2 = d1 - self.vol * math.sqrt(self.tau)

        return d1, d2

    def __Euro_call__(self):
        d1, d2 = self.__d1_d2__()
        Euro_NPV = self.S * math.exp(-self.q * self.tau) * self.__N__(d1) \
                   - math.exp(-self.r * self.tau) * self.K * self.__N__(d2)
        if self.tau == 0:
            Euro_NPV = max(self.S - self.K, 0)
        return Euro_NPV

    def __Euro_put__(self):
        d1, d2 = self.__d1_d2__()
        Euro_NPV = self.K * math.exp(-self.r * self.tau) * self.__N__(-d2) \
                   - self.S * math.exp(-self.q * self.tau) * self.__N__(-d1)

        if self.tau == 0:
            Euro_NPV = max(self.K - self.S, 0)
        return Euro_NPV

    def euro_theta_put(self):
        # !!TODO: calculation of theta is problematic
        d1, d2 = self.__d1_d2__()
        res = 0
        if self.option_type != "Put":
            raise Exception("For put only")
        res += self.r * self.K * math.exp(-self.r * self.tau) * ss.norm.cdf(-d2, 0, 1)
        res -= self.q * self.S * math.exp(-self.q * self.tau) * ss.norm.cdf(-d1, 0, 1)
        res -= self.vol * self.S / (2 * math.sqrt(self.tau)) * math.exp(-self.q * self.tau) * ss.norm.pdf(d1)
        return res

    def set_para(self, S, K, q, r, tau):
        self.S = S
        self.K = K
        self.r = r
        self.tau = tau
        self.q = q
