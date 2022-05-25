from model import SequentialNetwork, LinearLayer, LinearChannel, GaussianPrior, NetState, GaussBernoulliPrior, GaussianConvEnsemble, SparseGaussianConvEnsemble
from model import SequentialEnsemble, CompressiveSensing, GaussianPrior, LinearLayerEnsemble, GaussianConvEnsemble, MultiConvChannel, SparseMultiConvChannel
from model import PiecewiseLinearLayer, LinearChannel, ReLULinearLayerEnsemble, GaussianReLUConvEnsemble, AWGNChannel, GaussianAbsConvEnsemble
import pickle
import numpy
from arr import GET_NUMERICAL_LIB, to_cpu
import argparse
import os
xp = GET_NUMERICAL_LIB(device="default")
def amp_state_summary(act, state):
    mh = state['A'].mean()
    m = (act * state['h']).mean()
    mse = ( (act - state['h'])**2 ).mean()
    return {"mh": mh, "m": m, "err": mse}

def run(prior, alpha, channel_type, noise_std, exp_name, se=False, sparsity=0.25, dims=(500, 10, 3)):
    P, q, k = dims
    prior_ensemble = get_prior_ensemble(prior, noise_std, sparsity, dims)
    prior_network = prior_ensemble.problem_instance()

    if(len(prior_network.layers) > 0):
        N = prior_network.layers[0].shape[0]
    else:
        N = prior_network.prior.shape[0]
    M = int(alpha * N)


    if(channel_type == "cs"):
        channel_couplings = xp.random.normal(size=(M, N), scale=1 / xp.sqrt(N))
        activations = prior_network.sample()
        x0 = activations[0]
        y0 = channel_couplings @ x0 + xp.random.normal(size=y0.shape, scale=noise_std)
        A = LinearChannel(measurements=y0, noise_std=noise_std, couplings=channel_couplings)
        network = A.at(prior_network)
    elif(channel_type == "pr"):
        channel_couplings = xp.random.normal(size=(M, N), scale=1 / xp.sqrt(N))
        activations = prior_network.sample()
        x0 = activations[0]
        y0 = xp.abs(channel_couplings @ x0) + xp.random.normal(size=y0.shape, scale=noise_std)
        abs_layer = GaussianAbsConvEnsemble(alpha=alpha, inp_channels=P, image_shape=[q,], filter_size=k).sample()
        A = AWGNChannel(M, y0, noise_std=noise_std)
        network = prior_network.prepend(abs_layer).prepend(A)
    elif(channel_type == "conv-cs"):
        A_ens = SparseGaussianConvEnsemble(alpha=alpha, inp_channels=P, filter_size=k, image_shape=[q,], noise_std=noise_std)
        A_layer = A_ens.sample()
        activations = prior_network.sample()
        x0 = activations[0]
        y0 = A_layer.apply(x0)
        A = SparseMultiConvChannel(measurements=y0, noise_std=noise_std, filter=A_layer.filter, block_shape=[q,])
        network = A.at(prior_network)
    else:
        raise ValueError("Invalid channel type")

    print(f"Initialized network. Running {'se' if se else 'amp'} iterations for {prior} task at a={alpha}.")

    if(se):
        states = network.iterate_se(max_iter=75, early_stopping=False, verbose=True)
        with open(os.path.join('exp', f'{exp_name}.pkl'), 'wb+') as f_out:
            f_out.write(pickle.dumps(states))
    else:
        states = network.iterate_amp(max_iter=75, early_stopping=False, verbose=True)
        summary = [ [amp_state_summary(z, s) for z, s in zip(activations, state)] for state in states]
        with open(os.path.join('exp', f'{exp_name}_long.pkl'), 'wb+') as f_out:
            f_out.write(pickle.dumps({"states": states, "activations": activations}))
        with open(os.path.join('exp', f'{exp_name}.pkl'), 'wb+') as f_out:
            f_out.write(pickle.dumps(summary))

def get_prior_ensemble(prior, noise_std=0.01, sparsity=0.25, dims=(500, 10, 3)):
    P, q, k = dims

    layers = []
    latent_prior = None
    if(prior == "1-relu"):
        latent_prior = GaussianPrior(mean=0, std=1, dim=P * q)
        layers = [GaussianReLUConvEnsemble(alpha=2, inp_channels=P, image_shape=[q, ], filter_size=k)]
    elif (prior == "2-relu"):
        latent_prior = GaussianPrior(mean=0, std=1, dim=P * q)
        layers = [GaussianReLUConvEnsemble(alpha=1, inp_channels=2*P, image_shape=[q,], filter_size=k),
                  GaussianReLUConvEnsemble(alpha=2, inp_channels=P, image_shape=[q, ], filter_size=k)]
    elif (prior == "3-relu"):
        latent_prior = GaussianPrior(mean=0, std=1, dim=P * q)
        layers = 2*[GaussianReLUConvEnsemble(alpha=1, inp_channels=2*P, image_shape=[q,], filter_size=k)] + \
                 [GaussianReLUConvEnsemble(alpha=2, inp_channels=P, image_shape=[q, ], filter_size=k)]
    elif (prior == "1-linear"):
        latent_prior = GaussianPrior(mean=0, std=1, dim=P * q)
        layers = [GaussianConvEnsemble(alpha=2, inp_channels=P, image_shape=[q, ], filter_size=k, noise_std=noise_std)]
    elif (prior == "2-linear"):
        latent_prior = GaussianPrior(mean=0, std=1, dim=P * q)
        layers = [GaussianConvEnsemble(alpha=1, inp_channels=2*P, image_shape=[q,], filter_size=k, noise_std=noise_std),
                  GaussianConvEnsemble(alpha=2, inp_channels=P, image_shape=[q, ], filter_size=k, noise_std=noise_std)]
    elif (prior == "3-linear"):
        latent_prior = GaussianPrior(mean=0, std=1, dim=P * q)
        layers = 2*[GaussianConvEnsemble(alpha=1, inp_channels=2 * P, image_shape=[q,], filter_size=k, noise_std=noise_std)] + \
                 [GaussianConvEnsemble(alpha=2, inp_channels=P, image_shape=[q, ], filter_size=k, noise_std=noise_std)]
    elif (prior == "sparse"):
        latent_prior = GaussBernoulliPrior(mean=0, std=1, dim=P * q, sparsity=sparsity)

    return SequentialEnsemble(layers=layers, prior=latent_prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML AMP test code')
    parser.add_argument('--name', type=str, required=True, help='Name for this experiment, used to label output data.')
    parser.add_argument('--prior', type=str, required=True, help='Specify the prior to use for the experiment (sparse/1-relu/1-linear/2-relu/...).')
    parser.add_argument('-a', type=float, required=True, help='Measurement ratio (fraction D/P) for final linear sensing channel.')
    parser.add_argument('--channel', type=str, default='cs', help='Specify the activation channel (cs/conv-cs).')
    parser.add_argument('--noise_std', type=float, default=0.01, help='Noise standard deviation applied to measurements.')
    parser.add_argument('--sparsity', type=float, default=0.25, help='Specify the sparsity (fraction of nonzero entries) of priors with a sparsity parameters.')
    parser.add_argument('--se', action="store_true", default=False, help='If True, run SE instead of AMP.')
    parser.add_argument('--dims', nargs=3, type=int, default=[500, 10, 3], help='A list of integer values of [P, q, k] corresponding to the paper.')
    args = parser.parse_args()
    run(prior=args.prior,
        alpha=args.a,
        channel_type=args.channel,
        noise_std=args.noise_std,
        exp_name=args.name,
        sparsity=args.sparsity,
        dims=args.dims,
        se=args.se)