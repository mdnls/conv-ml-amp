from model import Prior, Channel
from arr import GET_NUMERICAL_LIB, to_cpu
xp = GET_NUMERICAL_LIB(device="default")
import numpy
import time

class SequentialNetwork():
    def __init__(self, layers, prior):
        '''
        Store an ordered list of layers and use the layers to produce new NetStates

        :param layers: an ordered list of layers.
        '''
        assert isinstance(prior, Prior), "prior must be a subclass of Prior"
        self.layers = layers
        self.prior = prior

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, item):
        return self.layers.__getitem__(item)

    def __setitem__(self, key, value):
        raise ValueError("SequentialNetwork layers are fixed after initialization.")

    def __delitem__(self, key):
        raise ValueError("SequentialNetwork layers are fixed after initialization.")

    def prepend(self, layer):
        return SequentialNetwork([layer] + self.layers, self.prior)

    def sample(self):
        activations = [self.prior.sample()]
        for layer in self.layers[::-1]:
            z = layer.apply(activations[0])
            activations = [z] + activations
        return activations

    def iterate_joint(self, true_values, early_stopping=True, max_iter=100, rtol=0.01):
        # stop once the largest relative fluctuations are smaller than rtol for 10 iterations
        states = [NetState(self)]
        se_states = [SEState(self)]
        true_overlaps = []
        i = 0
        stopct = 0
        while((i < max_iter) and (stopct < 10)):
            state, se_state, true_overlap = self.update_joint(states[-1], se_states[-1], true_values)
            states.append(state)
            se_states.append(se_state)
            true_overlaps.append(true_overlap)
            if(early_stopping and (len(states) > 2)):
                flucts = [
                    xp.linalg.norm(states[-1][i]['h'] - states[-2][i]['h']) / xp.linalg.norm(states[-1][i]['h'])
                    for i in range(len(states[-2])) if (states[-2][i]['h'] is not None)
                ]
                if(all([f < rtol for f in flucts])):
                    stopct += 1
                else:
                    stopct = 0
            i+=1
        state_params = [s.params for s in states]
        se_state_params = [s.params for s in se_states]
        return state_params, se_state_params, true_overlaps

    def iterate_amp(self, early_stopping=True, max_iter=100, rtol=0.01, verbose=False):
        # stop once the largest relative fluctuations are smaller than rtol for 10 iterations
        states = [NetState(self)]
        i = 0
        stopct = 0
        iter_start = time.time()
        iter_end = time.time()
        while((i < max_iter) and (stopct < 10)):
            states.append(self.update_amp(states[-1]))
            if(early_stopping and (len(states) > 2)):
                flucts = [
                    xp.linalg.norm(states[-1][i]['h'] - states[-2][i]['h']) / xp.linalg.norm(states[-1][i]['h'])
                    for i in range(len(states[-2])) if (states[-2][i]['h'] is not None)
                ]
                if(all([f < rtol for f in flucts])):
                    stopct += 1
                else:
                    stopct = 0
            i+=1
            if(verbose):
                iter_end = time.time()
                var_str = [f"L={i}: s={round(xp.mean(states[-1][i]['s']), 5)}, |h|={round(xp.linalg.norm(states[-1][i]['h']), 5)}" for i in range(len(states[-1]))]
                print(f"AMP iterate {i} in {iter_end - iter_start} secs. Variances: {', '.join(var_str)}")
                iter_start = iter_end
        state_params = [s.params for s in states]
        return state_params

    def iterate_se(self, early_stopping=True, max_iter=100, rtol=0.01, verbose=False):
        # stop once the largest relative fluctuations are smaller than rtol for 10 iterations
        states = [SEState(self)]
        i = 0
        stopct = 0
        while ((i < max_iter) and (stopct < 10)):
            states.append(self.update_se(states[-1]))
            if (early_stopping and (len(states) > 2)):
                flucts = [
                    xp.abs(states[-1][i]['m'] - states[-2][i]['m']) / states[-1][i]['m']
                    for i in range(len(states[-2])) if (states[-2][i]['m'] is not None)
                ]
                if (all([f < rtol for f in flucts])):
                    stopct += 1
                else:
                    stopct = 0
            i += 1
            if (verbose):
                print(f"SE iterate {i}")
        state_params = [s.params for s in states]
        return state_params

    def update_se(self, prev_state):
        state = SEState(self)
        L = len(state)

        p = state[0]['p']
        m_t = prev_state[0]['m']
        mh = self.layers[0].dual_magnetization(p, m_t)
        state[0]['mh'] = mh

        for l in range(1, L):
            p = state[l]['p']
            m_t = prev_state[l]['m']
            mh_lm1 = state[l-1]['mh']

            mh_l = self.layers[l].dual_magnetization(p, m_t, mh_lm1)
            state[l]['mh'] = mh_l

        mh_l = state[-1]['mh']
        p = state[-1]['p']
        m_tp1 = self.prior.magnetization(p, mh_l)
        state[-1]['m'] = m_tp1

        # compute m for layer L
        for l in range(0, L-1):
            p = state[l+1]['p']
            mh_l = state[l]['mh']
            m_lp1 = prev_state[l+1]['m']
            m_tp1 = self.layers[l+1].magnetization(p, m_lp1, mh_l)
            state[l]['m'] = m_tp1
        return state

    def update_joint(self, prev_state, prev_se_state, true_values):
        state = self.update_amp(prev_state)
        actual_overlaps = [
            {
                "m": xp.mean(state[l]['h'] * true_values[l]),
                "mh": xp.mean(state[l]['A'])
            }
            for l in range(len(state))
        ]
        se_state = self.update_se(prev_se_state)
        for l in range(len(state)):
            print(f"l={l}")
            print(f"Predicted overlap: {se_state[l]['m']}, mh={se_state[l]['mh']}")
            print(f"Actual overlap: {actual_overlaps[l]['m']}, mh={actual_overlaps[l]['mh']}")

        return state, se_state, actual_overlaps

    def update_amp(self, prev_state):
        state = NetState(self)
        L = len(state)

        # layer 1 update using the channel z_posterior
        h_t = prev_state[0]['h']
        s_t = prev_state[0]['s']
        g_tm1 = prev_state[0]['g']

        V = self.layers[0].sq_mm(s_t)
        w = self.layers[0].mm(h_t) - V * g_tm1
        g, g_pr = self.layers[0].z_posterior(V, w)
        A = -self.layers[0].T_sq_mm(g_pr)
        B = self.layers[0].T_mm(g) + A * h_t
        state[0] = {
            "V": V,
            "A": A,
            "g": g,
            "g_pr": g_pr,
            "h": None,
            "s": None,
            "w": w,
            "B": B
        }

        for l in range(1, L):
            layer = self.layers[l]
            h_t = prev_state[l]['h']
            s_t = prev_state[l]['s']
            g_tm1 = prev_state[l]['g']

            V = layer.sq_mm(s_t)
            w = layer.mm(h_t) - V * g_tm1
            state[l]['V'] = V
            state[l]['w'] = w

            # at time t
            A_lm1 = state[l-1]['A']
            B_lm1 = state[l-1]['B']
            g, g_pr = layer.z_posterior(A_lm1, B_lm1, V, w)
            state[l]['g'] = g
            state[l]['g_pr'] = g_pr

            A_l = -layer.T_sq_mm(g_pr)
            B_l = layer.T_mm(g) + A_l * h_t
            state[l]['A'] = A_l
            state[l]['B'] = B_l

        # layer L+1 update using prior h_posterior

        A_l, B_l = state[-1]['A'], state[-1]['B']
        h_tp1, s_tp1 = self.prior.h_posterior(A_l, B_l)
        state[-1]['h'] = h_tp1
        state[-1]['s'] = s_tp1

        for l in range(0, L-1):
            A_l, B_l = state[l]['A'], state[l]['B']
            V_lp1, w_lp1 = state[l+1]['V'], state[l+1]['w']
            h_tp1, s_tp1 = self.layers[l+1].h_posterior(A_l, B_l, V_lp1, w_lp1)
            state[l]['h'] = h_tp1
            state[l]['s'] = s_tp1
        return state

class SEState():
    '''
    Datastructure containing AMP state evolution parameters.
    '''
    def __init__(self, net):
        self.network = net

        overlaps = [net.prior.self_overlap()]
        for layer in self.network.layers[::-1]:
            overlaps = [layer.output_self_overlap(overlaps[0])] + overlaps
        params = [{
            "p": overlap,
            "m": 0,
            "mh": 1/overlap,
        }
        for overlap in overlaps[1:]]
        self.params = params

    def __len__(self):
        return len(self.network)

    def __getitem__(self, item):
        return self.params.__getitem__(item)

    def __setitem__(self, key, value):
        self.params.__setitem__(key, value)

class NetState():
    '''
    An datastructure which contains AMP parameters corresponding to a fixed network, fixed quenched couplings,
        at a specific timestep/amp iteration.
    '''
    def __init__(self, net):
        self.network = net
        overlaps = [net.prior.self_overlap()]
        for layer in self.network.layers[::-1]:
            overlaps = [layer.output_self_overlap(overlaps[0])] + overlaps
        self.params = [
            {
                "V": xp.ones((layer.shape[0], 1)), # variance of Z
                "A": xp.ones((layer.shape[1], 1)), # first moment of H
                "g": xp.zeros((layer.shape[0], 1)), # (standardized) first moment of Z, first derivative of layer partition
                "g_pr": xp.ones((layer.shape[0], 1)), # derivative of g
                "h": xp.zeros((layer.shape[1], 1)), # first moment of H, corresponding to index t+1
                "s": overlap * xp.ones((layer.shape[1], 1)), # variance of H, corresponding to index t+1
                "w": xp.zeros((layer.shape[0], 1)), # onsager correction of Z local field
                "B": xp.zeros((layer.shape[1], 1))  # onsager correction of H local field
            }
            for overlap, layer in zip(overlaps[1:], self.network.layers)
        ]

    def __len__(self):
        return len(self.network)

    def __getitem__(self, item):
        return self.params.__getitem__(item)

    def __setitem__(self, key, value):
        self.params.__setitem__(key, value)

