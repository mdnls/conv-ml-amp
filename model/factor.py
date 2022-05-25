from model.piecewise_linear import LinearRegion, gaussian_measure, gaussian_measure_2d, phi_0, phi_1, phi_2
from arr import GET_NUMERICAL_LIB, GET_DEVICE, to_cpu
device = GET_DEVICE()
xp = GET_NUMERICAL_LIB(device=device)

if(device == "gpu"):
    import cupyx.scipy as sp
    import cupyx.scipy.ndimage
    import cupyx.scipy.stats
    import cupyx.scipy.special
else:
    import scipy as sp
    import scipy.ndimage
    import scipy.stats
    import scipy.special

from scipy.special import softmax, expit

def sigmoid(x):
    return expit(x)

def N(x):
    return sp.stats.norm.pdf(x)

def C(x):
    return sp.stats.norm.cdf(x)


class Factor():
    def __init__(self, couplings, noise_std=0.001):
        '''
        Represents a SISO linear channel with AWGN and fixed input couplings.

        NOTE: children may *not* have couplings, their methods may use fast
        :param couplings: couplings of this linear channel
        :param noise_std: std of noise applied to outputs of this channel
        '''
        self._couplings = couplings
        self._sqr_couplings = self._couplings**2
        self.noise_std = noise_std
        self.shape = couplings.shape

    def mm(self, h):
        return self._couplings @ h

    def sq_mm(self, h):
        return self._sqr_couplings @ h

    def T_mm(self, z):
        return self._couplings.T @ z

    def T_sq_mm(self, z):
        return self._sqr_couplings.T @ z

    def output_self_overlap(self, input_self_overlap):
        return input_self_overlap + self.noise_std**2

    def apply(self, z):
        return self._couplings @ z + xp.random.normal(size=(self.shape[0], 1), scale=self.noise_std)

    def at(self, network):
        return network.prepend(self)


class Channel(Factor):
    def __init__(self, couplings, noise_std=0.001):
        super(Channel, self).__init__(couplings, noise_std)

class Prior():
    def __init__(self):
        pass

class LinearLayer(Factor):
    def z_posterior(self, A, B, V, w):
        Vinv = 1 / V
        A_D = self.noise_std ** (-2)
        S = (A + A_D) / (Vinv * A + A_D * A + A_D * Vinv)
        vb = A_D / (A + A_D)

        avg_z = S * ((Vinv * w) + vb * B)
        g = Vinv * (avg_z - w)
        g_pr = Vinv * (S * Vinv - 1)
        return g, g_pr

    def h_posterior(self, A, B, V, w):
        Vinv = 1 / V
        A_D = self.noise_std ** (-2)
        S = (Vinv + A_D) / (Vinv * A + A_D * A + A_D * Vinv)
        vw = A_D / (Vinv + A_D)

        h = S * (vw * (w / V) + B)
        return h, S

    def dual_magnetization(self, p, m, mh):
        a = self.shape[0] / self.shape[1]
        Vinv = 1 / (p - m)
        A_D = self.noise_std ** (-2)
        S = (mh + A_D) / (A_D * mh + A_D * Vinv + mh * Vinv)

        g_pr = Vinv * (S * Vinv - 1)
        mh_l = -a * g_pr
        return mh_l

    def magnetization(self, p, m, mh):
        Vinv = 1 / (p - m)
        A_D = self.noise_std ** (-2)
        S = (Vinv + A_D) / (A_D * mh + A_D * Vinv + mh * Vinv)
        vb = A_D / (Vinv + A_D)
        m_l = S * (m * vb * Vinv + mh * (p + self.noise_std ** 2))
        return m_l

class LinearChannel(Channel):
    def __init__(self, measurements, couplings, noise_std=0.001):
        super(LinearChannel, self).__init__(couplings, noise_std)
        self.y = measurements

    def z_posterior(self, V, w):
        Vinv = 1 / V
        A_D = self.noise_std ** (-2)
        S = 1 / (Vinv + A_D)
        avg_z = S * ((w / V) + (A_D * self.y))

        g = Vinv * (avg_z - w)
        g_pr = Vinv * (S * Vinv - 1)
        return g, g_pr

    def h_posterior(self, A, B, V, w):
        raise ValueError("Channels cannot compute h posterior because h represents the output of a layer")

    def output_self_overlap(self, input_self_overlap):
        return input_self_overlap + self.noise_std**2

    def dual_magnetization(self, p, m):
        alpha = self.shape[0] / self.shape[1]
        Vinv = 1 / (p - m)
        A_D = self.noise_std ** (-2)
        S = 1 / (Vinv + A_D)
        g_pr = Vinv * (S * Vinv - 1)
        return - alpha * g_pr

class AWGNChannel(Channel):
    def __init__(self, input_dim, measurements, noise_std=0.001):
        super(AWGNChannel, self).__init__(xp.eye(input_dim), noise_std)
        self.y = measurements
        self.dim = input_dim

    def z_posterior(self, V, w):
        Vinv = 1 / V
        A_D = self.noise_std ** (-2)
        S = 1 / (Vinv + A_D)
        avg_z = S * ((w / V) + (A_D * self.y))
        g = Vinv * (avg_z - w)
        g_pr = Vinv * (S * Vinv - 1)
        return g, g_pr

    def h_posterior(self, A, B, V, w):
        raise ValueError("Channels cannot compute h posterior because h represents the output of a layer")

    def output_self_overlap(self, input_self_overlap):
        return input_self_overlap + self.noise_std**2

    def dual_magnetization(self, p, m):
        alpha = self.shape[0] / self.shape[1]
        Vinv = 1 / (p - m)
        A_D = self.noise_std ** (-2)
        S = 1 / (Vinv + A_D)
        g_pr = Vinv * (S * Vinv - 1)
        return - alpha * g_pr

class GaussBernoulliPrior(Prior):
    def __init__(self, mean, std, sparsity, dim):
        # sparsity is percent of nonzero coordinates
        super(GaussBernoulliPrior, self).__init__()
        self.mean = mean
        self.std = std
        self.prec = std**(-2)
        self.sparsity = sparsity
        self.slo = -xp.log(self.sparsity / (1 - self.sparsity)) # sparsity log odds
        self.shape = [dim, dim]

    def h_posterior(self, A, B):
        # see 'E.1.7 Sparse variable' p51 of Tree Approximate Message Passing, Aubin, Baker, Krzakala, Zdeborova
        A_eff = A + self.prec
        B_eff = B + (self.prec * self.mean)
        z = B_eff**2 / (2*A_eff) - self.mean**2 / (2 * self.prec) + 0.5 * xp.log(1 / A_eff) - self.slo
        def sig(x):
            return 1 / (1 + xp.exp(-x))
        sig_z = sig(z)
        h = (B_eff/A_eff) * sig_z
        s = (sig_z/A_eff) + (B_eff/A_eff)**2 * (sig_z) * (1-sig_z)
        return h, s

    def sample(self):
        mask = (xp.random.uniform(size=(self.shape[0], 1)) < self.sparsity)
        return mask * xp.random.normal(size=(self.shape[0], 1), loc=self.mean, scale=self.std)

    def self_overlap(self):
        return self.sparsity * (self.mean**2 + self.std**2)

    def magnetization(self, p, mh):
        def overlap_func(b):
            def sig(x):
                return 1 / (1 + xp.exp(-x))
            sig_z = sig(b**2 / (2*(mh+1)) - self.slo + 0.5 * xp.log(1/ (mh+1)))
            return sig_z / (mh+1) + (b / (mh+1))**2 * sig_z * (1-sig_z)

        return mh * self.sparsity * gaussian_measure(0, xp.sqrt(mh + mh**2), overlap_func)

class GaussianPrior(Prior):
    def __init__(self, mean, std, dim):
        super(GaussianPrior, self).__init__()
        assert xp.array(mean).flatten().shape[0] == 1, "Mean must be a scalar"
        assert xp.array(std).flatten().shape[0] == 1, "Mean must be a scalar"
        self.mean = mean
        self.std = std
        self.dim = dim
        self.shape = [dim, dim]

    def h_posterior(self, A, B):
        B_h_eff = B + self.mean / self.std**2
        A_h_eff = A + self.std**(-2)
        h, s = B_h_eff/A_h_eff, 1/A_h_eff
        return h, s

    def sample(self):
        S = self.mean + self.std * xp.random.normal(size=(self.dim, 1))
        return S

    def self_overlap(self):
        return self.std**2 + self.mean**2

    def magnetization(self, p, mh):
        return self.mean**2 / (1 + self.std**2 * mh) + (mh*self.std**2) / (mh + self.std**(-2))

class PiecewiseLinearLayer(Factor):
    def __init__(self, couplings, lines, monte_carlo=False):
        '''
        Apply a deterministic piecewise linear channel function
        :param couplings: matrix W of coupling coefficients defining h = Wz
        :param lines: a list of piecewise linear regions of the form [zmin, zmax, intercept, slope]
        :param monte_carlo: if True, estimate SE quantities via monte carlo
        '''
        super(PiecewiseLinearLayer, self).__init__(couplings, noise_std=0)
        self.regions = [LinearRegion(**region) for region in lines]
        self.monte_carlo = monte_carlo
        self.n_mc_samples = 20000

    def scalar_apply(self, z):
        return sum(seg.sample(z) for seg in self.regions)

    def apply(self, h):
        z = self.mm(h)
        return sum(seg.sample(z) for seg in self.regions)

    def merge_estimates(self, rs, vs, As):
        ps = softmax(As, axis=0)
        r = sum(p*r for p, r in zip(ps, rs))
        Dr = sum(
            ps[i]*ps[j]*(rs[i] - rs[j])**2
            for i in range(len(self.regions))
            for j in range(i+1, len(self.regions))
        )
        v = sum(p*v for p, v in zip(ps, vs)) + Dr
        return r, v

    def z_posterior(self, A, B, V, w):
        Bz, Az = w/V, 1/V
        # in TRAMP, the backward/forward conventions are reversed compared to ML-amp
        rs = [region.backward_mean(Az, Bz, A, B) for region in self.regions]
        vs = [region.backward_variance(Az, Bz, A, B)
              for region in self.regions]
        As = [region.log_partitions(Az, Bz, A, B) for region in self.regions]
        r, v = self.merge_estimates(rs, vs, As)
        return (r - w)/V, (v/V - 1)/V

    def h_posterior(self, A, B, V, w):
        Bz, Az = w/V, 1/V
        rs = [region.forward_mean(Az, Bz, A, B) for region in self.regions]
        vs = [region.forward_variance(Az, Bz, A, B)
              for region in self.regions]
        As = [region.log_partitions(Az, Bz, A, B) for region in self.regions]
        r, v = self.merge_estimates(rs, vs, As)
        return r, v

    def output_self_overlap(self, input_self_overlap):
        taus = [region.second_moment(input_self_overlap) for region in self.regions]
        ps = [region.proba_tau(input_self_overlap) for region in self.regions]
        tau_x = sum(p * tau for p, tau in zip(ps, taus))
        return tau_x

    def beliefs_measure(self, az, ax, tau_z, f):
        mu = sum(
            region.beliefs_measure(az, ax, tau_z, f) for region in self.regions
        )
        return mu

    def dual_magnetization(self, p, m, mh):
        if(self.monte_carlo):
            samples = xp.random.normal(size=(3, self.n_mc_samples))
            m1 = xp.sqrt(m)
            m2 = xp.sqrt(p - m)
            A = xp.array([[m1, 0, 0],
                          [m1, m2, 0],
                          [0, 0, 0],
                          [0, 0, 1]])
            samples = A @ samples
            samples[2, :] = self.scalar_apply(samples[1, :])
            samples[3, :] = mh * samples[2, :] + xp.sqrt(mh) * samples[3, :]
            g_w = self.z_posterior(mh, samples[3, :], p - m, samples[0, :])[1]
            alpha = self.shape[0] / self.shape[1]
            return xp.mean(- alpha * g_w)
        else:
            alpha = self.shape[0] / self.shape[1]
            Vz = p - m
            def v_func(bz, bx):
                rz = bz * Vz
                _, v = self.z_posterior(mh, bx, Vz, rz)
                return v
            v_est = self.beliefs_measure(1/Vz, mh, p, v_func)
            return - alpha * v_est

    def magnetization(self, p, m, mh):
        h_self_overlap = self.output_self_overlap(p)
        if(self.monte_carlo):
            samples = xp.random.normal(size=(3, self.n_mc_samples))
            m1 = xp.sqrt(m)
            m2 = xp.sqrt(p - m)
            A = xp.array([[m1, 0, 0],
                              [m1, m2, 0],
                              [0, 0, 0],
                              [0, 0, 1]])
            samples = A @ samples
            samples[2, :] = self.scalar_apply(samples[1, :])
            samples[3, :] = mh * samples[2, :] + xp.sqrt(mh) * samples[3, :]
            h_hat = self.h_posterior(mh, samples[3, :], p - m, samples[0, :])[0]
            m = xp.mean(samples[2, :] * h_hat)
            if(m > h_self_overlap):
                return (h_self_overlap - 1e-10)
            else:
                return m
        else:
            Vz = p-m

            def v_func(bz, bx):
                rz = bz * Vz
                _, s = self.h_posterior(mh, bx, Vz, rz)
                return s
            v_est = self.beliefs_measure(1/Vz, mh, p, v_func)
            return h_self_overlap - v_est

class AbsLinearLayer(LinearLayer):
    def __init__(self, couplings):
        raise NotImplementedError('This implementation is bugged and cannot be run.')
        super(AbsLinearLayer, self).__init__(couplings, noise_std=0)

    def scalar_apply(self, z):
        return xp.abs(z)

    def apply(self, h):
        z = self.mm(h)
        return xp.abs(z)

    def h_posterior(self, A, B, V, w):
        # estimate x from x = abs(z)
        Az, Bz = 1/V, w/V
        a = A + Az
        x_pos = (B + Bz) / xp.sqrt(a)
        x_neg = (B - Bz) / xp.sqrt(a)
        delta = phi_0(x_pos) - phi_0(x_neg)
        sigma_pos = sigmoid(+delta)
        sigma_neg = sigmoid(-delta)
        r_pos = phi_1(x_pos) / xp.sqrt(a)
        r_neg = phi_1(x_neg) / xp.sqrt(a)
        v_pos = phi_2(x_pos) / a
        v_neg = phi_2(x_neg) / a
        rx = sigma_pos * r_pos + sigma_neg * r_neg
        Dx = (r_pos - r_neg)**2
        v = sigma_pos * sigma_neg * Dx + sigma_pos * v_pos + sigma_neg * v_neg
        vx = xp.ones_like(rx) * xp.mean(v)
        return rx, vx

    def z_posterior(self, A, B, V, w):
        Az, Bz = 1/V, w/V
        a = A + Az
        x_pos = (B + Bz) / xp.sqrt(a)
        x_neg = (B - Bz) / xp.sqrt(a)
        delta = phi_0(x_pos) - phi_0(x_neg)
        sigma_pos = sigmoid(+delta)
        sigma_neg = sigmoid(-delta)
        r_pos = + phi_1(x_pos) / xp.sqrt(a) # NB: + phi'(x_pos)
        r_neg = - phi_1(x_neg) / xp.sqrt(a) # NB: - phi'(x_neg)
        v_pos = phi_2(x_pos) / a
        v_neg = phi_2(x_neg) / a
        rz = sigma_pos * r_pos + sigma_neg * r_neg
        Dz = (r_pos  - r_neg)**2
        v = sigma_pos * sigma_neg * Dz + sigma_pos * v_pos + sigma_neg * v_neg
        vz = xp.ones_like(rz) * xp.mean(v)
        return (rz - w)/V, (vz/V - 1)/V

    def beliefs_measure(self, az, ax, tau_z, f):
        u_eff = xp.maximum(0, az * tau_z - 1)
        sz_eff = xp.sqrt(az * u_eff)

        def f_pos(bz, bx):
            a = ax + az
            x_pos = (bx + bz) / xp.sqrt(a)
            return C(x_pos) * f(bz, bx)

        def f_neg(bz, bx):
            a = ax + az
            x_neg = (bx - bz) / xp.sqrt(a)
            return N(x_neg) * f(bz, bx)

        if ax==0 or u_eff==0:
            sx_eff = xp.sqrt(ax * (ax * tau_z + 1))
            mu_pos = mu_neg = gaussian_measure_2d(0, sz_eff, 0, sx_eff, f_pos)
        else:
            cov_pos = xp.array([
                [ax * (ax * tau_z + 1), +ax * u_eff],
                [+ax * u_eff, az * u_eff]
            ])
            cov_neg = xp.array([
                [ax * (ax * tau_z + 1), -ax * u_eff],
                [-ax * u_eff, az * u_eff]
            ])
            mu_pos = gaussian_measure_2d_full(cov_pos, 0, f_pos)
            mu_neg = gaussian_measure_2d_full(cov_neg, 0, f_neg)
        return mu_pos + mu_neg

    def dual_magnetization(self, p, m, mh):
        alpha = self.shape[0] / self.shape[1]
        Vz = p - m
        def v_func(bz, bx):
            rz = bz * Vz
            _, v = self.z_posterior(mh, bx, Vz, rz)
            return v
        v_est = self.beliefs_measure(1/Vz, mh, p, v_func)
        return - alpha * v_est

    def magnetization(self, p, m, mh):
        h_self_overlap = self.output_self_overlap(p)
        Vz = p-m
        def v_func(bz, bx):
            rz = bz * Vz
            _, s = self.h_posterior(mh, bx, Vz, rz)
            return s
        v_est = self.beliefs_measure(1/Vz, mh, p, v_func)
        return h_self_overlap - v_est

class ReLULinearLayer(PiecewiseLinearLayer):
    def __init__(self, couplings, monte_carlo=False):
        lines = [{"zmin": -xp.inf, "zmax": 0, "x0": 0, "slope": 0},
                 {"zmin": 0, "zmax": xp.inf, "x0": 0, "slope": 1}]
        super(PiecewiseLinearLayer).__init__(couplings, lines=lines, monte_carlo=monte_carlo)

