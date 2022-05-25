'''

class PiecewiseLinearChannel(Factor):
    def __init__(self, couplings, partition_boundaries, lines):
        super(PiecewiseLinearChannel, self).__init__(couplings, noise_std=0)
        self.partition_boundaries = partition_boundaries
        self.lines = lines

        partition_intervals = [(None, partition_boundaries[0])] + \
                              list(zip(self.partition_boundaries, self.partition_boundaries[1:])) + \
                              [(partition_boundaries[-1], None)]

        self.lines_data = [(x, y, x_min, x_max) for (x, y), (x_min, x_max) in zip(lines, partition_intervals)]

    def _truncated_gaussian_moments(self, ai, bi, x_min, x_max):
        z_min = np.sqrt(ai) * (x_min - bi/ai)
        z_max = np.sqrt(ai) * (x_max - bi/ai)
        ri = (b/a) - (N(z_max) - N(z_min))/(np.sqrt(a) * (C(z_max) - C(z_min)))
        vi = (1/a) * (1 - (z_max * N(z_max) - z_min * N(z_min)) / (C(z_max) - C(z_min))
                      - ( (N(z_max) - N(z_min)) / (C(z_max) - C(z_min)) )**2
                      )
        Ai = bi**2 / (2 * ai) + (1/2) * np.log(2 * np.pi / ai) + np.log(C(z_max) - C(z_min))
        return ri, vi, Ai

    def evaluate(self, x):
        for l in range(len(self.partition_boundaries)):
            if(self.partition_boundaries[l] > x):
                return self.lines[0] * x + self.lines[1]

    def z_posterior(self, V, w):
        Bz, Az = w/V, 1/V
        moments_per_lines = [
            self._truncated_gaussian_moments(ai=Az + y**2 * A,
                                             bi=Bz + y*(B - A*x),
                                             x_min=x_min,
                                             x_max=x_max)
            for (x, y, x_min, x_max) in self.lines_data
        ]
        R_l_z, V_l_z, A_l = zip(*moments_per_lines)
        x_l, y_l = zip(*self.lines)

        P_l = sp.special.softmax([
            A_l[i] + B * x_l[i] - 0.5 * A * (x_l[i]**2)
            for i in range(len(L))
        ])
        rz = sum([R_l_z[i] * P_l[i] for i in range(len(L))])
        vz = sum([V_l_z[i] * P_l[i] for i in range(len(L))]) + \
             sum([V_l_z[i] * V_l_z[j] * (R_l_z[i] - R_l_z[j])**2 for i in range(len(L)) for j in range(i)])
        return (rz - w)/V, (vz - 1)/V

    def h_posterior(self, A, B, V, w):
        Bz, Az = w/V, 1/V
        moments_per_lines = [
            self._truncated_gaussian_moments(ai=Az + y**2 * A,
                                             bi=Bz + y*(B - A*x),
                                             x_min=x_min,
                                             x_max=x_max)
            for (x, y, x_min, x_max) in self.lines_data
        ]
        R_l_z, V_l_z, A_l = zip(*moments_per_lines)
        x_l, y_l = zip(*self.lines)
        R_l_h = [x_l[i] + y_l[i] * R_l_z[i] for i in range(len(L))]
        V_l_h = [y_l[i]**2 * V_l_z[i] for i in range(len(L))]

        P_l = sp.special.softmax([
            A_l[i] + B * x_l[i] - 0.5 * A * (x_l[i]**2)
            for i in range(len(L))
        ])

        rh = sum([R_l_h[i] * P_l[i] for i in range(len(L))])
        vh = sum([V_l_h[i] * P_l[i] for i in range(len(L))]) + \
             sum([V_l_h[i] * V_l_h[j] * (R_l_h[i] - R_l_h[j])**2 for i in range(len(L)) for j in range(i)])
        return rh, vh


    def output_self_overlap(self, input_self_overlap):
        return (1/2) * input_self_overlap

    def dual_magnetization(self, p, m):
        alpha = self.shape[0] / self.shape[1]
        Vinv = 1 / (p - m)
        A_D = self.noise_std ** (-2)
        S = 1 / (Vinv + A_D)

        g_pr = Vinv * (S * Vinv - 1)
        return - alpha * g_pr

'''