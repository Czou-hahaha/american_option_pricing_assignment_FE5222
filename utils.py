# /*
#  * @Author: Jingzhe Luo, Yidan Zou, Jiali Liu
#  * @Date: 2023-11-09 23:01:16
#  * @Last Modified by:   Jingzhe Luo
#  * @Last Modified time: 2023-11-14 03:29:32
#  */

import math as math
import scipy.stats as ss
import numpy as np
import numpy.polynomial.legendre as legendre
import matplotlib.pyplot as plt


def d1_(t, z, r, q, sigma):
    # if t < 1e-5:
    #     raise Exception("Too short maturity remaining!")
    res = (np.log(z) + (r - q + 1 / 2 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return res


def d2_(t, z, r, q, sigma):
    return d1_(t, z, r, q, sigma) - sigma * np.sqrt(t)


def pdf_d1(t, z, r, q, sigma):
    return ss.norm.pdf(d1_(t, z, r, q, sigma))


def pdf_d2(t, z, r, q, sigma):
    return ss.norm.pdf(d2_(t, z, r, q, sigma))


def cdf_d1(t, z, r, q, sigma):
    return ss.norm.cdf(d1_(t, z, r, q, sigma))


def cdf_d2(t, z, r, q, sigma):
    return ss.norm.cdf(d2_(t, z, r, q, sigma))


def phi_(x):
    return ss.norm.cdf(x)


def cheby_to_a(q):
    """
    @Description: Generate necessary parameters for chebyshev interpolation on interval [-1,1].
    """
    # !!Notice ordered from right points to left points
    num = np.shape(q)[1] - 1
    a = np.zeros(np.shape(q))
    a = a + 1 / num * (q[0, 0] + q[0, -1] * np.cos(np.pi * np.arange(0, num + 1)))
    i = np.arange(0, num + 1).reshape((-1, 1))
    k = np.arange(0, num + 1).reshape((1, -1))
    ik = np.dot(i, k) / num
    cos_ = np.cos(np.pi * ik)

    a = a + np.dot(q[0, 1:num], 2 / num * cos_[1:num, :])

    return a.reshape((1, -1))


def cheby_to_value(a, z):
    """
    @Description: value the function value at set points given interpolation  parameters.
    """
    x = -z
    m = np.shape(a)[1] - 1
    b1 = 0
    b2 = 0
    b0 = a[0, -1] / 2
    for j in range(m - 1, -1, -1):
        b1, b2 = b0, b1
        b0 = a[0, j] + 2 * x * b1 - b2
    return 0.5 * (b0 - b2)


def cheby_points_z(num_points):
    """
    Generate cheby points on [-1,1] for given number
    """
    i = np.arange(0, num_points + 1)
    z = -np.cos(i * np.pi / num_points)
    return np.reshape(z, (1, -1))


def tau_to_z(max_tau, tau_arr):
    # m = tau_arr
    # mm = np.reshape(max_tau, (1, -1))
    return np.sqrt(4 * tau_arr / np.reshape(max_tau, (1, -1))) - 1


def z_to_tau(max_tau, z_arr):
    return max_tau / 4 * np.power((1 + z_arr), 2)


def b_to_h(x_base, b):
    return np.power(np.log(b / x_base), 2)


def h_to_b(x_base, h):
    return x_base * np.exp(np.sqrt(h))


def cheby_points_tau(tau_max, num_points):
    i = np.arange(0, num_points + 1).reshape((1, -1))
    z = -np.cos(i * np.pi / num_points)
    x_ = np.sqrt(tau_max) / 2 * (1 + z)
    tau_ = np.power(x_, 2)
    return tau_


def gauss_std_integrate_points(num_points):
    nodes, weights = legendre.leggauss(num_points)
    return nodes, weights


def gauss_integrate(integrand_fun, num_points, tau):
    # num_points = len(tau)
    nodes, weights = legendre.leggauss(num_points)
    ans = []
    if len(tau) != 1:
        for i in range(len(tau)):
            ans[i] = weights * integrand_fun(nodes, tau[i])
    else:
        ans = weights * integrand_fun(nodes, tau)
    return ans


def runge_func(x):
    return np.exp(2 * x)


def tau_vec_to_matrix(integral_std_y, inter_tau):
    y = 1-np.power(1+integral_std_y, 2)/4
    return np.dot(np.reshape(y, (-1, 1)), np.reshape(inter_tau, (1, -1)))


if __name__ == "__main__":
    err_ls = list()
    for n in range(3, 20):
        z_points = cheby_points_z(n)
        f_value = runge_func(z_points)
        para_a = cheby_to_a(f_value)
        e_z = (np.arange(2*n+1)/2/n - 0.5) * 2
        ff = runge_func(e_z)
        fit_f = cheby_to_value(para_a, e_z)
        err_ls.append(np.mean(np.abs((fit_f - ff)/ff)))
        plt.plot(e_z, ff, "--")
        plt.plot(e_z, fit_f)
        plt.show()
        c = 1
    print(err_ls)

    # m = 10
    # node, weight = gauss_std_integrate_points(m)
    # print(np.shape(node))


