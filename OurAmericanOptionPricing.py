# /*
#  * @Author: Jingzhe Luo, Yidan Zou, Jiali Liu
#  * @Date: 2023-11-09 23:01:16 
#  * @Last Modified by:   Jingzhe Luo 
#  * @Last Modified time: 2023-11-13 04:11:27
#  */
import time

import numpy as np
from EuropeanPricing import EuropeanOption
import QdPlus
import utils


class Pricing(EuropeanOption):
    parameter_set: bool

    def __init__(self, n=10, m=10, l=10) -> None:
        super().__init__()
        self.update_value = None
        self.integral_h = None
        self.zz = None
        self.h = None
        self.f_prime = None
        self.d_prime_value = None
        self.n_prime_value = None
        self.d_value = None
        self.n_value = None
        self.f = None
        self.integral_b_value = None
        self.inter_para_a = None
        self.price = None
        self.option_price = None
        self.early_exercise_premium = None
        self.euro_price = None
        self.maturity = None
        self.curr_t = None
        self.parameter_set = False
        self.n = n
        self.m = m
        self.l = l

    def set_parameters(self, t0=0, t_stop=1, volatility=0.2, free_rate=0.05, div_rate=0, strike=100, option_type='Call',
                       curr_price=100) -> None:
        self.option_type = option_type
        assert self.option_type in ['Call', "Put"], "Option type should be either Put or Call!"
        self.curr_t = t0
        self.maturity = t_stop
        assert t0 <= t_stop, "Maturity date should be no earlier than current time!"
        self.tau = t_stop - t0
        self.q = div_rate
        self.vol = volatility
        self.K = strike
        self.r = free_rate
        self.S = curr_price
        self.option_price = 0
        self.early_exercise_premium = 0
        self.parameter_set = True

    def solver(self):
        if not self.parameter_set:
            print('Please set parameters before pricing')
            raise Exception("Parameters not set yet!")

        if self.tau < 1 / 400:
            self.price = max(self.K - self.S, 0) if self.option_type == "Put" else max(self.S - self.K, 0)
            return self.price

        if self.option_type == "Call" and self.q == 0:
            self.euro_price = self._europ_solver()
            self.price = self.euro_price
            return self.price

        elif self.option_type == "Put":
            self._set_cal_para()
            self.euro_price = self._europ_solver()
            self.early_exercise_premium = self._american_put_premium()
            self.price = self.early_exercise_premium + self.euro_price
            return self.price
        else:
            self._exchange_para()
            self._set_cal_para()
            # self.early_exercise_premium = self._american_put_premium()
            # self.euro_price = self._europ_solver()
            self.price = self._american_put_premium() + self._europ_solver()
            self._exchange_para()

            return self.price

    def _american_put_premium(self):
        ans = 0
        self._solve_boundary()

        self.h = utils.b_to_h(self.x_base, self.boundary)
        self.inter_para_a = utils.cheby_to_a(self.h)
        self.integral_price_b_value = utils.h_to_b(self.x_base, utils.cheby_to_value(self.inter_para_a, self.integral_price_z))

        v1 = self.r * self.K
        v1 = v1 * np.exp(-self.r * (self.tau - self.integral_price_tau)) * \
             (1 - utils.cdf_d2(self.tau - self.integral_price_tau, self.S / self.integral_price_b_value, self.r, self.q,
                               self.vol)) * (1 + self.integral_std_price_y)/2*self.tau
        ans += np.dot(self.integral_std_price_w.reshape((1, -1)), v1.reshape((-1, 1)))
        v2 = self.q * self.S * np.exp(-self.q * (self.tau - self.integral_price_tau)) * \
             (1 - utils.cdf_d1(self.tau - self.integral_price_tau, self.S / self.integral_price_b_value, self.r, self.q,
                               self.vol)) * (1 + self.integral_std_price_y)/2*self.tau

        ans -= np.dot(self.integral_std_price_w.reshape((1, -1)), v2.reshape((-1, 1)))
        return ans

    def _solve_boundary(self) -> None:
        # chebyshev: 
        # integral: output
        # init QD+
        solve_qd = QdPlus.QdInitial(self.r, self.q, self.S, self.K, self.vol, self.inter_tau)
        self.boundary[0, 1:] = solve_qd.init_price()
        # self.inter_para_a = utils.cheby_to_a(self.h)
        j = 0
        while j < self.m:
            self.iter_boundary()
            j += 1
        brea_here = 1

    def _exchange_para(self) -> None:
        rr = self.q
        qq = self.r
        spot = self.K
        ss = self.S
        self.r = rr
        self.q = qq
        self.K = ss
        self.S = spot
        if self.option_type == "Call":
            self.option_type = "Put"
        else:
            self.option_type = "Call"

    def _set_cal_para(self) -> None:
        self.boundary = np.zeros((1, self.n + 1))
        if self.r != 0:
            if self.q != 0:
                self.boundary[0, 0] = self.K * min(1,
                                                   self.r / self.q)
            else:
                self.boundary[0, 0] = self.K
            self.x_base = self.boundary[0, 0] / self.K
        else:
            self.boundary[0, 0] = 0
            self.x_base = 0
        self.inter_z = utils.cheby_points_z(self.n)
        self.inter_tau = utils.cheby_points_tau(self.tau, self.n)
        self.inter_tau_use = np.reshape(self.inter_tau[0, 1:], (1, -1))
        self.integral_std_y, self.integral_std_w = utils.gauss_std_integrate_points(self.l)
        self.integral_std_y = self.integral_std_y.reshape((-1, 1))
        self.integral_std_w = np.reshape(self.integral_std_w, (1, -1))
        self.integral_tau_matrix = utils.tau_vec_to_matrix(self.integral_std_y, self.inter_tau[0, 1:])
        self.integral_z_matrix = utils.tau_to_z(self.tau, self.integral_tau_matrix)
        self.integral_std_price_y, self.integral_std_price_w = utils.gauss_std_integrate_points(2 * self.l)
        self.integral_std_price_y = self.integral_std_price_y.reshape((-1, 1))
        self.integral_price_tau = self.tau - self.tau * np.power(self.integral_std_price_y + 1, 2) / 4
        self.integral_price_z = np.sqrt(self.integral_price_tau / self.tau * 4) - 1

        # self.value_points = utils.

    def iter_boundary(self) -> None:
        eta = 0.5
        self.h = utils.b_to_h(self.x_base, self.boundary)
        self.inter_para_a = utils.cheby_to_a(self.h)
        self.integral_h = utils.cheby_to_value(self.inter_para_a, self.integral_z_matrix)
        self.integral_b_value = utils.h_to_b(self.x_base,
                                             self.integral_h)
        self.zz = self.boundary[0, 1:] / self.integral_b_value
        self.cal_f()
        self.cal_f_prime()
        self.update_value = eta * (self.boundary[0, 1:] - self.f) / (self.f_prime - 1)
        self.boundary[0, 1:] = self.boundary[0, 1:] + eta * (self.boundary[0, 1:] - self.f) / (self.f_prime - 1)
        self.h = utils.b_to_h(self.x_base, self.boundary)

    def cal_f(self):
        self.cal_n()
        self.cal_d()
        self.f = self.K * np.exp(-(self.r - self.q) * self.inter_tau_use) * self.n_value / self.d_value

    def cal_n(self):
        self.n_value = utils.pdf_d2(self.inter_tau_use, self.boundary[0, 1:] / self.K, self.r, self.q, self.vol) \
                       / (self.vol * np.sqrt(self.inter_tau_use))

        self.n_value = self.n_value + self.r * self.cal_k3()

    def cal_d(self):
        self.d_value = self.q * (self.cal_k1() + self.cal_k2())
        self.d_value = self.d_value + utils.cdf_d1(self.inter_tau_use, self.boundary[0, 1:] / self.K,
                                                   self.r, self.q, self.vol)
        self.d_value = self.d_value + utils.pdf_d1(self.inter_tau_use, self.boundary[0, 1:] / self.K,
                                                   self.r, self.q, self.vol) / (self.vol * np.sqrt(self.inter_tau_use))

    def cal_k3(self):
        integrand = 1 / self.vol * np.exp(-self.r * (self.inter_tau_use - self.integral_tau_matrix)) * \
                    utils.pdf_d2(self.inter_tau_use - self.integral_tau_matrix,
                                 self.zz, self.r, self.q, self.vol)
        return np.exp(self.r * self.inter_tau_use) * np.sqrt(self.inter_tau_use) * \
               np.dot(self.integral_std_w, integrand)

    def cal_k1(self):
        c = 1
        integrand = np.exp(-self.q * (self.inter_tau_use - self.integral_tau_matrix)) * \
                    (1 + self.integral_std_y) * \
                    utils.cdf_d1(self.inter_tau_use - self.integral_tau_matrix,
                                 self.zz, self.r, self.q, self.vol)
        return 0.5 * self.inter_tau_use * np.exp(self.q * self.inter_tau_use) * \
               np.dot(self.integral_std_w, integrand)

    def cal_k2(self):
        integrand = 1 / self.vol * np.exp(-self.q * (self.inter_tau_use - self.integral_tau_matrix)) * \
                    utils.pdf_d1(self.inter_tau_use - self.integral_tau_matrix,
                                 self.zz, self.r, self.q, self.vol)
        return np.sqrt(self.inter_tau_use) * \
               np.exp(self.q * self.inter_tau_use) * np.dot(self.integral_std_w, integrand)

    def cal_f_prime(self):
        self.cal_n_prime()
        self.cal_d_prime()
        self.f_prime = self.K * np.exp(-(self.r - self.q) * self.inter_tau_use) * \
                       (self.n_prime_value / self.d_value - self.d_prime_value * self.n_value / np.power(self.d_value,
                                                                                                         2))

    def cal_n_prime(self):
        self.n_prime_value = -utils.d2_(self.inter_tau_use, self.boundary[0, 1:] / self.K, self.r, self.q, self.vol) * \
                             utils.pdf_d2(self.inter_tau_use, self.boundary[0, 1:] / self.K, self.r, self.q,
                                          self.vol) / (self.inter_tau_use * self.boundary[0, 1:] * self.vol ** 2)
        self.n_prime_value = self.n_prime_value - self.r * self.cal_n_prime_integral()

    def cal_d_prime(self):
        self.d_prime_value = - utils.d2_(self.inter_tau_use,
                                         self.boundary[0,
                                         1:] / self.K,
                                         self.r, self.q,
                                         self.vol) * \
                             utils.pdf_d1(self.inter_tau_use, self.boundary[0, 1:] / self.K, self.r, self.q, self.vol) / \
                             (self.inter_tau_use * self.boundary[0, 1:] * self.vol ** 2)
        self.d_prime_value = self.d_prime_value - self.q * self.cal_d_prime_integral()

    def cal_n_prime_integral(self):
        integrand = np.exp(self.r * self.integral_tau_matrix) * utils.d2_(
            self.inter_tau_use - self.integral_tau_matrix,
            self.zz, self.r, self.q, self.vol) * utils.pdf_d2(
            self.inter_tau_use - self.integral_tau_matrix,
            self.zz, self.r, self.q, self.vol) / self.boundary[0, 1:] / self.vol ** 2 / (
                            1 + self.integral_std_y) * 2

        return np.dot(self.integral_std_w, integrand)

    def cal_d_prime_integral(self):
        integrand = np.exp(self.q * self.integral_tau_matrix)
        integrand = integrand * utils.d2_(
            self.inter_tau_use - self.integral_tau_matrix,
            self.zz, self.r, self.q, self.vol)
        integrand = integrand * utils.pdf_d1(
            self.inter_tau_use - self.integral_tau_matrix,
            self.zz, self.r, self.q, self.vol) / self.vol ** 2 / self.boundary[0, 1:] / (
                            1 + self.integral_std_y) * 2

        return np.dot(self.integral_std_w, integrand)


if __name__ == "__main__":
    t, T, vol, rate, d_r, s, o_type, price = (0, 3, 0.2, 0.05, 0.05, 100, "Put", 80)
    my_pricer = Pricing(n=10, m=5, l=24)
    my_pricer.set_parameters(t, T, vol, rate, d_r, s, o_type, price)
    tic = time.time()
    print("price is", my_pricer.solver()[0])
    print("time used", time.time() - tic)
    c = 1
