from arr import GET_NUMERICAL_LIB, to_cpu
xp = GET_NUMERICAL_LIB(device="default")
import numpy as np
from scipy.integrate import quad, dblquad

'''
This code is largely a copied, GPU enabled version of TRAMP code:

Tree-AMP: Compositional Inference with Tree Approximate Message Passing
Baker, Aubin, Krzakala, Zdeborová
https://arxiv.org/abs/2004.01571

with associated code:
https://github.com/sphinxteam/tramp/blob/master/tramp/channels/activation/piecewise_linear_channel.py
'''


from scipy.special import erf, erfc, erfcx
import scipy.stats



def norm_cdf(x):
    return scipy.stats.norm.cdf(x)

def norm_pdf(x):
    return scipy.stats.norm.pdf(x)

def gaussian_measure_2d(m1, s1, m2, s2, f):
    def integrand(x2, x1):
        return norm_pdf(x1) * norm_pdf(x2) * f(m1 + s1 * x1, m2 + s2 * x2)
    integral = dblquad(integrand, -20, 20, -20, 20)[0]
    return integral

def is_pos_def(x):
    return np.all(np.linalg.eigvalsh(x) > 0)

def gaussian_measure_2d_full(mean, cov, f):
    if not is_pos_def(cov):
        logger.warn(f"cov={cov} not positive definite")
    L = xp.linalg.cholesky(cov)

    def integrand(x2, x1):
        y1, y2 = L @ [x1, x2] + mean
        return norm_pdf(x1) * norm_pdf(x2) * f(y1, y2)
    integral = dblquad(integrand, -20, 20, -20, 20)[0]
    return integral

def gaussian_measure(m, s, f, z_min=-100, z_max=100):
    def integrand(x):
        return norm_pdf(x) * f(m + s * x)
    integral = quad(integrand, z_min, z_max)[0]
    return integral

def switch(x, y):
    "Switch x and y values to have np.abs(x) <= np.abs(y)"
    x_new = xp.where(xp.abs(x) > xp.abs(y), y, x)
    y_new = xp.where(xp.abs(x) > xp.abs(y), x, y)
    return x_new, y_new

def log_Phi_nonzero(x):
    return xp.log(0.5 * erfcx(-x / xp.sqrt(2)))-0.5*x**2

def log_Phi(x):
    y = xp.zeros_like(x, dtype=float)
    nonzero = (x < 30)
    y[nonzero] = log_Phi_nonzero(x[nonzero])
    return y

def F0_inf(x, y):
    "Return F0(x, y) where y is inf"
    return xp.log(erfcx(xp.sign(y)*x)) - x**2

def F0_close(x, y):
    e = y - x
    L = (
        - x*e
        + (1/6)*(x**2 - 2)*e**2
        - (1/180)*(x**4 + 2*x**2 - 8)
        + xp.log(2*e/xp.sqrt(xp.pi))
    )
    return L - x**2

def F0_neg(x, y):
    D = xp.exp(x**2 - y**2)
    L = xp.log(xp.abs(
        D * erfcx(-y) - erfcx(-x)
    ))
    return L - x**2

def F0_pos(x, y):
    D = xp.exp(x**2 - y**2)
    L = xp.log(xp.abs(
        erfcx(x) - D * erfcx(y)
    ))
    return L - x**2

def F0_other(x, y):
    return xp.log(xp.abs(erf(y) - erf(x)))

def F0(x, y, thresh=1e-7):
    "Computes log|erf(y) - erf(x)|"
    x, y = switch(x, y)
    assert (xp.abs(x) <= xp.abs(y)).all()
    # conditions
    inf = xp.isinf(y)
    close = ~inf & (xp.abs(x - y) <= thresh)
    neg = (x < 0) & (y < 0)
    pos = (x > 0) & (y > 0)
    other = ~(neg | pos)
    neg = ~inf & ~close & neg
    pos = ~inf & ~close & pos
    other = ~inf & ~close & other
    # F0 values
    F = xp.zeros_like(x, dtype=float)
    F[inf] = F0_inf(x[inf], y[inf])
    F[close] = F0_close(x[close], y[close])
    F[neg] = F0_neg(x[neg], y[neg])
    F[pos] = F0_pos(x[pos], y[pos])
    F[other] = F0_other(x[other], y[other])

    return F

def F1_inf(x, y):
    "Return F1(x, y) where y is inf"
    return xp.sign(y)/erfcx(xp.sign(y)*x)

def F1_close(x, y):
    e = y - x
    return xp.sqrt(xp.pi) * (
        x
        + (1/2)*e
        - (1/6)*e**2
        - (1/12)*e**3
        + (1/90)*x*(x**2+1.)*e**4
    )

def F1_neg(x, y):
    D = xp.exp(x**2 - y**2)
    return (1 - D) / (D * erfcx(-y) - erfcx(-x))

def F1_pos(x, y):
    D = xp.exp(x**2 - y**2)
    return (1 - D) / (erfcx(x) - D * erfcx(y))

def F1_other(x, y):
    D = xp.exp(x**2 - y**2)
    return xp.exp(-x**2) * (1 - D) / (erf(y) - erf(x))

def F1(x, y, thresh=1e-7):
    "Computes (exp(-x^2) - exp(-y^2)) / (erf(y) - erf(x))"
    x, y = switch(x, y)
    assert (xp.abs(x) <= xp.abs(y)).all()
    # conditions
    inf = xp.isinf(y)
    close = ~inf & (xp.abs(x - y) <= thresh)
    neg = (x < 0) & (y < 0)
    pos = (x > 0) & (y > 0)
    other = ~(neg | pos)
    neg = ~inf & ~close & neg
    pos = ~inf & ~close & pos
    other = ~inf & ~close & other
    # F1 values
    F = xp.zeros_like(x, dtype=float)
    F[inf] = F1_inf(x[inf], y[inf])
    F[close] = F1_close(x[close], y[close])
    F[neg] = F1_neg(x[neg], y[neg])
    F[pos] = F1_pos(x[pos], y[pos])
    F[other] = F1_other(x[other], y[other])

    return F

def F2_inf(x, y):
    "Return F2(x, y) where y is inf"
    return xp.sign(y)*x/erfcx(xp.sign(y)*x)

def F2_close(x, y):
    "Taylor expansion of F2(x, x+e)"
    e = y - x
    return xp.sqrt(xp.pi) * (
        x**2 - 1/2
        + x*e
        - (1/3)*(x**2-1)*e**2
        - (1/3)*x*e**3
        + (1/90)*(2*x**4 + 3*x**2 - 8)*e**4
    )


def F2_neg(x, y):
    D = xp.exp(x**2 - y**2)
    return (x - D * y) / (D * erfcx(-y) - erfcx(-x))


def F2_pos(x, y):
    D = xp.exp(x**2 - y**2)
    return (x - D * y) / (erfcx(x) - D * erfcx(y))


def F2_other(x, y):
    D = xp.exp(x**2 - y**2)
    return xp.exp(-x**2) * (x - D * y) / (erf(y) - erf(x))


def F2(x, y, thresh=1e-7):
    "Computes (x*exp(-x^2) - y*exp(-y^2)) / (erf(y) - erf(x))"
    x, y = switch(x, y)
    assert (xp.abs(x) <= xp.abs(y)).all()
    # conditions
    inf = xp.isinf(y)
    close = ~inf & (xp.abs(x - y) <= thresh)
    neg = (x < 0) & (y < 0)
    pos = (x > 0) & (y > 0)
    other = ~(neg | pos)
    neg = ~inf & ~close & neg
    pos = ~inf & ~close & pos
    other = ~inf & ~close & other
    # F2 values
    F = xp.zeros_like(x, dtype=float)
    F[inf] = F2_inf(x[inf], y[inf])
    F[close] = F2_close(x[close], y[close])
    F[neg] = F2_neg(x[neg], y[neg])
    F[pos] = F2_pos(x[pos], y[pos])
    F[other] = F2_other(x[other], y[other])

    return F


def G0(x, y):
    "Computes log|Phi(y) - Phi(x)|"
    return xp.log(0.5) + F0(x/xp.sqrt(2), y/xp.sqrt(2))


def G1(x, y):
    "Computes [N(x) - N(y)] / [Phi(y) - Phi(x)]"
    return xp.sqrt(2/xp.pi) * F1(x/xp.sqrt(2), y/xp.sqrt(2))


def G2(x, y):
    "Computes [y*N(y) - x*N(x)] / [Phi(y) - Phi(x)]"
    return (2/xp.sqrt(xp.pi)) * F2(x/xp.sqrt(2), y/xp.sqrt(2))


def G0_inf(x, s):
    "Computes G0(x, +inf) or G0(x, -inf)"
    # return xp.log(0.5) + F0_inf(x/xp.sqrt(2), s)
    return log_Phi(-s*x)


def G1_inf(x, s):
    "Computes G1(x, +inf) or G1(x, -inf)"
    return xp.sqrt(2/xp.pi) * F1_inf(x/xp.sqrt(2), s)


def G2_inf(x, s):
    "Computes G2(x, +inf) or G2(x, -inf)"
    return (2/xp.sqrt(xp.pi)) * F2_inf(x/xp.sqrt(2), s)


def truncated_normal_mean(r0, v0, zmin, zmax):
    "Mean of N(z | r0 v0) delta_[zmin, zmin](z)"
    assert zmin < zmax
    s0 = xp.sqrt(v0)
    ymin = (zmin - r0) / s0
    ymax = (zmax - r0) / s0
    if (zmax == +xp.inf):
        g1 = G1_inf(ymin, +1)
    elif (zmin == -xp.inf):
        g1 = G1_inf(ymax, -1)
    else:
        g1 = G1(ymin, ymax)
    r = r0 + s0 * g1
    return r


def truncated_normal_var(r0, v0, zmin, zmax):
    "Variance of N(z | r0 v0) delta_[zmin, zmin](z)"
    assert zmin < zmax
    s0 = xp.sqrt(v0)
    ymin = (zmin - r0) / s0
    ymax = (zmax - r0) / s0
    if (zmax == +xp.inf):
        g1 = G1_inf(ymin, +1)
        g2 = G2_inf(ymin, +1)
    elif (zmin == -xp.inf):
        g1 = G1_inf(ymax, -1)
        g2 = G2_inf(ymax, -1)
    else:
        g1 = G1(ymin, ymax)
        g2 = G2(ymin, ymax)
    v = v0 * (1. + g2 - g1**2)
    return v


def truncated_normal_log_proba(r0, v0, zmin, zmax):
    "Log proba of z in [zmin, zmin] for N(z | r0 v0)"
    assert zmin < zmax
    s0 = xp.sqrt(v0)
    ymin = (zmin - r0) / s0
    ymax = (zmax - r0) / s0
    if (zmax == +xp.inf):
        g0 = G0_inf(ymin, +1)
    elif (zmin == -xp.inf):
        g0 = G0_inf(ymax, -1)
    else:
        g0 = G0(ymin, ymax)
    return g0

def truncated_normal_proba(r0, v0, zmin, zmax):
    "Proba of z in [zmin, zmin] for N(z | r0 v0)"
    assert zmin < zmax
    s0 = xp.sqrt(v0)
    ymin = -xp.inf if zmin == -xp.inf else (zmin - r0) / s0
    ymax = +xp.inf if zmax == +xp.inf else (zmax - r0) / s0
    p = norm_cdf(ymax) - norm_cdf(ymin)
    return p

def truncated_normal_logZ(r0, v0, zmin, zmax):
    "Log Partition of N(z | r0 v0) delta_[zmin, zmin](z)"
    g0 = truncated_normal_log_proba(r0, v0, zmin, zmax)
    logZ = 0.5*xp.log(2*xp.pi*v0) + 0.5*r0**2/v0 + g0
    return logZ


class LinearRegion():
    def __init__(self, zmin, zmax, x0, slope):
        assert zmin < zmax
        self.zmin = zmin
        self.zmax = zmax
        self.x0 = x0
        self.slope = slope

    def x(self, z):
        return self.x0 + self.slope*z

    def sample(self, Z):
        # zero outside of the region
        X = self.x(Z) * (self.zmin <= Z) * (Z < self.zmax)
        return X

    def get_r0_v0(self, az, bz, ax, bx):
        a = az + self.slope**2 * ax
        b = bz + self.slope * (bx - ax * self.x0)
        r0 = b / a
        v0 = 1 / a
        return r0, v0

    def backward_mean(self, az, bz, ax, bx):
        r0, v0 = self.get_r0_v0(az, bz, ax, bx)
        rz = truncated_normal_mean(r0, v0, self.zmin, self.zmax)
        return rz

    def backward_variance(self, az, bz, ax, bx):
        r0, v0 = self.get_r0_v0(az, bz, ax, bx)
        vz = truncated_normal_var(r0, v0, self.zmin, self.zmax)
        return vz

    def forward_mean(self, az, bz, ax, bx):
        rz = self.backward_mean(az, bz, ax, bx)
        rx = self.slope * rz + self.x0
        return rx

    def forward_variance(self, az, bz, ax, bx):
        vz = self.backward_variance(az, bz, ax, bx)
        vx = self.slope**2 * vz
        return vx

    def log_partitions(self, az, bz, ax, bx):
        "Element-wise log_partition"
        r0, v0 = self.get_r0_v0(az, bz, ax, bx)
        trunc_logZ = truncated_normal_logZ(r0, v0, self.zmin, self.zmax)
        logZ = trunc_logZ - 0.5*ax*self.x0**2 + bx*self.x0
        return logZ

    def second_moment(self, tau_z):
        rz = truncated_normal_mean(0, tau_z, self.zmin, self.zmax)
        vz = truncated_normal_var(0, tau_z, self.zmin, self.zmax)
        rx = self.slope * rz + self.x0
        vx = self.slope**2 * vz
        tau_x = rx**2 + vx
        return tau_x

    def proba_tau(self, tau_z):
        p = truncated_normal_proba(0, tau_z, self.zmin, self.zmax)
        return p

    def proba_ab(self, az, bz, ax, bx):
        r0, v0 = self.get_r0_v0(az, bz, ax, bx)
        p = truncated_normal_proba(r0, v0, self.zmin, self.zmax)
        return p

    def beliefs_measure(self, az, ax, tau_z, f):
        u_eff = np.maximum(0, az * tau_z - 1)
        mean_x = ax*self.x0

        def integrand(bz, bx):
            return self.proba_ab(az, bz, ax, bx) * f(bz, bx)

        if ax == 0 or u_eff == 0 or self.slope == 0:
            sz_eff = np.sqrt(az * u_eff)
            sx_eff = np.sqrt(ax * (self.slope**2 * ax * tau_z + 1))
            mu = gaussian_measure_2d(0, sz_eff, mean_x, sx_eff, integrand)
        else:
            cov = np.array([
                [az * u_eff, self.slope * ax * u_eff],
                [self.slope * ax * u_eff, ax * (self.slope**2 * ax * tau_z + 1)]
            ])
            mean = np.array([0, mean_x])
            mu = gaussian_measure_2d_full(mean, cov, integrand)
        return mu


def log_norm_cdf_prime(x):
    "Computes (log Phi)'(x) = N(x)/Phi(x)"
    d = np.sqrt(2 * np.pi) * 0.5 * erfcx(-x / np.sqrt(2))
    return 1. / d

def phi_0(x):
    return np.log(0.5 * erfcx(-x / np.sqrt(2)))

def phi_1(x):
    y = log_norm_cdf_prime(x)
    return x + y

def phi_2(x):
    y = log_norm_cdf_prime(x)
    return 1 - y * (x + y)

