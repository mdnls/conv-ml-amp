from model.factor import LinearChannel, LinearLayer, PiecewiseLinearLayer
from model.conv import MultiConvLayer, PiecewiseLinearMultiConvLayer, SparseMultiConvLayer, ReLUMultiConvLayer, AbsMultiConvLayer
from model.network import SequentialNetwork
from arr import GET_NUMERICAL_LIB, to_cpu
import numpy as np
xp = GET_NUMERICAL_LIB(device="default")

class SequentialEnsemble():
    def __init__(self, layers, prior):
        self.layers = layers
        self.prior = prior

    def problem_instance(self):
        '''
        Sample a network from this ensemble representing the PGM form of a problem instance
        :return: SequentialNetwork
        '''
        sample_layers = []
        for layer in self.layers:
            sample_layers.append(layer.sample())
        return SequentialNetwork(layers=sample_layers, prior=self.prior)

class Ensemble():
    def __init__(self, alpha, inp_dim, noise_std):
        self.alpha = alpha
        self.inp_dim = inp_dim
        self.noise_std = noise_std

    def sample(self):
        raise NotImplementedError("Children implement this method.")

class ReLULinearLayerEnsemble(Ensemble):
    def __init__(self, alpha, inp_dim, monte_carlo=False):
        super(ReLULinearLayerEnsemble, self).__init__(alpha=alpha, inp_dim=inp_dim, noise_std=0)
        self.monte_carlo = monte_carlo

    def sample(self):
        outp_dim = int(self.alpha * self.inp_dim)
        couplings = xp.random.normal(size=(outp_dim, self.inp_dim), scale=1/xp.sqrt(self.inp_dim))
        return PiecewiseLinearLayer(couplings,
                                    lines=[{"zmin": -xp.inf, "zmax": 0, "x0": 0, "slope": 0},
                                        {"zmin": 0, "zmax": xp.inf, "x0": 0, "slope": 1}],
                                    monte_carlo=self.monte_carlo)

class LinearLayerEnsemble(Ensemble):
    def __init__(self, alpha, inp_dim, noise_std):
        super(LinearLayerEnsemble, self).__init__(alpha, inp_dim, noise_std)

    def sample(self):
        outp_dim = int(self.alpha * self.inp_dim)
        couplings = xp.random.normal(size=(outp_dim, self.inp_dim), scale=1/xp.sqrt(self.inp_dim))
        return LinearLayer(couplings, noise_std=self.noise_std)

class GaussianConvEnsemble(Ensemble):
    def __init__(self, alpha, inp_channels, image_shape, filter_size, noise_std):
        outp_channels = int(alpha * inp_channels)
        inp_dim = inp_channels * np.prod(image_shape)
        super(GaussianConvEnsemble, self).__init__(alpha, inp_dim, noise_std)
        self.inp_channels = inp_channels
        self.outp_channels = outp_channels
        self.image_shape = image_shape
        self.order = len(image_shape)
        self.filter_size = filter_size

    def sample(self):
        nnz_per_row = self.filter_size**(self.order) * self.inp_channels
        filter = xp.random.normal(size=[self.outp_channels, self.inp_channels] + self.order * [self.filter_size], scale=1/np.sqrt(nnz_per_row))
        return MultiConvLayer(filter=filter, block_shape=self.image_shape, noise_std=self.noise_std)

class SparseGaussianConvEnsemble(Ensemble):
    def __init__(self, alpha, inp_channels, image_shape, filter_size, noise_std):
        outp_channels = int(alpha * inp_channels)
        inp_dim = inp_channels * np.prod(image_shape)
        super(SparseGaussianConvEnsemble, self).__init__(alpha, inp_dim, noise_std)
        self.inp_channels = inp_channels
        self.outp_channels = outp_channels
        self.image_shape = image_shape
        self.order = len(image_shape)
        self.filter_size = filter_size

    def sample(self):
        nnz_per_row = self.filter_size**(self.order) * self.inp_channels
        filter = xp.random.normal(size=[self.outp_channels, self.inp_channels] + self.order * [self.filter_size], scale=1/np.sqrt(nnz_per_row))
        return SparseMultiConvLayer(filter=filter, block_shape=self.image_shape, noise_std=self.noise_std)


class GaussianAbsConvEnsemble(Ensemble):
    def __init__(self, alpha, inp_channels, image_shape, filter_size):
        outp_channels = int(alpha * inp_channels)
        inp_dim = inp_channels * np.prod(image_shape)
        super(GaussianAbsConvEnsemble, self).__init__(alpha, inp_dim, noise_std=0)
        self.inp_channels = inp_channels
        self.outp_channels = outp_channels
        self.image_shape = image_shape
        self.order = len(image_shape)
        self.filter_size = filter_size

    def sample(self):
        nnz_per_row = self.filter_size**(self.order) * self.inp_channels
        filter = xp.random.normal(size=[self.outp_channels, self.inp_channels] + self.order * [self.filter_size], scale=1/np.sqrt(nnz_per_row))
        return AbsMultiConvLayer(filter=filter, block_shape=self.image_shape)

class GaussianReLUConvEnsemble(Ensemble):
    def __init__(self, alpha, inp_channels, image_shape, filter_size, monte_carlo=False):
        outp_channels = int(alpha * inp_channels)
        inp_dim = inp_channels * np.prod(image_shape)
        super(GaussianReLUConvEnsemble, self).__init__(alpha, inp_dim, noise_std=0)
        self.inp_channels = inp_channels
        self.outp_channels = outp_channels
        self.image_shape = image_shape
        self.order = len(image_shape)
        self.filter_size = filter_size
        self.monte_carlo = monte_carlo

    def sample(self):
        nnz_per_row = self.filter_size**(self.order) * self.inp_channels
        filter = xp.random.normal(size=[self.outp_channels, self.inp_channels] + self.order * [self.filter_size], scale=1/np.sqrt(nnz_per_row))
        return ReLUMultiConvLayer(filter=filter, block_shape=self.image_shape, monte_carlo=self.monte_carlo)


