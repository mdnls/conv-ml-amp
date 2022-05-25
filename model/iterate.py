from model import LinearChannel
import numpy as np
from arr import GET_NUMERICAL_LIB, to_cpu
xp = GET_NUMERICAL_LIB(device="default")

class Iterator():
    def __init__(self, net_ensemble):
        self.net_ensemble = net_ensemble

    def evaluate(self, alphas, repeats, max_iter, early_stopping, rtol):
        raise NotImplementedError("Children implement me!")

class ConvolutionalSensing(Iterator):
    def __init__(self, network, image_shape, filter_size, noise_std):
        super(ConvolutionalSensing, self).__init__(network)
        assert (network[0].shape[0] // np.prod(image_shape)) == (network[0].shape[0] / np.prod(image_shape)), "Image size must divide network output size"
        self.noise_std = noise_std
        self.image_shape = image_shape
        self.filter_size = filter_size
        self.order = len(image_shape)

    def evaluate_se(self, alphas, max_iter=200, early_stopping=True, rtol=0.01, verbose=False):
        mse_per_alpha = []
        for a in alphas:
            network = self.net_ensemble.problem_instance()

            N = network[0].shape[0] // np.prod(self.image_shape)
            M = int(a * N)

            activations = network.sample()
            raise NotImplementedError("Below: find a way to evaluate y without instantiating the MCC, since you need y to instantiate.")
            filter = xp.random.normal(size=[M, N] + self.order*[self.filter_size], scale=1/xp.sqrt(N * self.filter_size**self.order))
            x = activations[0]
            y = couplings @ x + xp.random.normal(size=(M, 1), scale=self.noise_std)

            channel = MultiConvChannel(measurements=y, filter=couplings, block_shape=self.image_shape, noise_std=self.noise_std)
            signal_model = channel.at(network)

            states = signal_model.iterate_se(early_stopping, max_iter, rtol)
            final_state = states[-1]
            mse = to_cpu(final_state[-1]['p'] - final_state[-1]['m'])

            mse_per_alpha.append(to_cpu(mse))
            if(verbose):
                print(f"At a={round(a, 3)}, SE predicts MSE {round(mse_per_alpha[-1], 5)}")
        return mse_per_alpha

    def evaluate_amp(self, alphas, repeats=1, max_iter=200, early_stopping=True, rtol=0.01, verbose=False):
        avg_mse_per_alpha = []
        for a in alphas:
            mses = []
            for k in range(repeats):
                network = self.net_ensemble.problem_instance()

                N = network[0].shape[0]
                M = int(a * N)

                activations = network.sample()
                couplings = xp.random.normal(size=(M, N), scale=1/xp.sqrt(N))

                x = activations[0]
                y = couplings @ x + xp.random.normal(size=(M, 1), scale=self.noise_std)
                channel = LinearChannel(measurements=y, couplings=couplings, noise_std=self.noise_std)
                signal_model = channel.at(network)

                states = signal_model.iterate_amp(early_stopping, max_iter, rtol)
                final_state = states[-1]
                mse = to_cpu(xp.mean((final_state[-1]['h'] - activations[-1])**2))
                if(mse > 10):
                    print("Outlier!")
                else:
                    mses.append(mse)
            avg_mse_per_alpha.append(np.mean(mses))
            if(verbose):
                print(f"At a={round(a, 3)}, AMP MSE of {round(avg_mse_per_alpha[-1], 5)} (avg over {repeats} repeats)")
        return avg_mse_per_alpha

class CompressiveSensing(Iterator):
    def __init__(self, network, noise_std):
        super(CompressiveSensing, self).__init__(network)
        self.noise_std = noise_std

    def evaluate_se(self, alphas, max_iter=200, early_stopping=True, rtol=0.01, verbose=False):
        mse_per_alpha = []
        states_per_alpha = []
        for a in alphas:
            network = self.net_ensemble.problem_instance()
            if(len(network) > 0):
                N = network[0].shape[0]
            else:
                N = network.prior.shape[0]
            M = int(a * N)

            activations = network.sample()
            couplings = xp.random.normal(size=(M, N), scale=1/xp.sqrt(N))
            x = activations[0]
            y = couplings @ x + xp.random.normal(size=(M, 1), scale=self.noise_std)

            channel = LinearChannel(measurements=y, couplings=couplings, noise_std=self.noise_std)
            signal_model = channel.at(network)

            states = signal_model.iterate_se(early_stopping, max_iter, rtol)
            final_state = states[-1]
            mse = to_cpu(final_state[-1]['p'] - final_state[-1]['m'])

            mse_per_alpha.append(to_cpu(mse))
            states_per_alpha.append(states)
            if(verbose):
                print(f"At a={round(a, 3)}, SE predicts MSE {round(mse_per_alpha[-1], 5)}")
        return mse_per_alpha, states_per_alpha

    def evaluate_amp(self, alphas, repeats=1, max_iter=200, early_stopping=True, rtol=0.01, verbose=False):
        avg_mse_per_alpha = []
        states_per_alpha = []
        for a in alphas:
            mses = []
            rep_states = []
            for k in range(repeats):
                network = self.net_ensemble.problem_instance()

                N = network[0].shape[0]
                M = int(a * N)

                activations = network.sample()
                couplings = xp.random.normal(size=(M, N), scale=1/xp.sqrt(N))

                x = activations[0]
                y = (couplings @ x) + xp.random.normal(size=(M, 1), scale=self.noise_std)
                channel = LinearChannel(measurements=y, couplings=couplings, noise_std=self.noise_std)
                signal_model = channel.at(network)

                states = signal_model.iterate_amp(early_stopping, max_iter, rtol)
                final_state = states[-1]
                mse = to_cpu(xp.mean((final_state[-1]['h'] - activations[-1])**2))
                if(mse > 1.5):
                    print("Outlier!")
                else:
                    mses.append(mse)
                    rep_states.append(states)
            avg_mse_per_alpha.append(np.mean(mses))
            states_per_alpha.append(rep_states)
            if(verbose):
                print(f"At a={round(a, 3)}, AMP MSE of {round(avg_mse_per_alpha[-1], 5)} (avg over {repeats} repeats)")
        return avg_mse_per_alpha, states_per_alpha



