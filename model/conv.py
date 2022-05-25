from model.factor import LinearLayer, LinearChannel, PiecewiseLinearLayer, AbsLinearLayer
import numpy as np
from arr import GET_NUMERICAL_LIB, to_cpu
xp = GET_NUMERICAL_LIB(device="default")
import scipy as sp
import scipy.ndimage
from numba import jit

@jit(nopython=True, parallel=True)
def fast_1d_multiconv(z, conv_filter, q):
    D, P, k = conv_filter.shape
    result = np.zeros(shape=(D, q))
    for i in range(D):
        for j in range(P):
            for l in range(k):
                for s in range(q):
                    result[i, s] += z[j, (l + s - (k-1)//2) % q] * conv_filter[i, j, l]
    return result

@jit(nopython=True, parallel=True)
def fast_1d_T_multiconv(z, conv_filter, q):
    D, P, k = conv_filter.shape
    result = np.zeros(shape=(P, q))
    for i in range(D):
        for j in range(P):
            for s in range(q):
                for l in range(k):
                    result[j, s] += z[i, (l + s - (k-1)//2) % q] * conv_filter[i, j, k-l-1]
    return result


class PiecewiseLinearMultiConvLayer(PiecewiseLinearLayer):
    def __init__(self, filter, block_shape, lines, monte_carlo=False):
        super(PiecewiseLinearMultiConvLayer, self).__init__(filter, lines, monte_carlo)
        m, n = filter.shape[:2]
        filter_shape = filter.shape[2:]

        self.macro_shape = (m, n)
        self.block_shape = list(block_shape)
        self.block_order = len(block_shape)
        self.filter_shape = filter_shape

        self.shape = (m * np.prod(block_shape), n * np.prod(block_shape))

        if(self.block_order > 24):
            raise ValueError(f"Input data blocks have tensor order {self.block_order} > 24 which will \
            break einstein sums used in this implementation.")

        self.filter = filter
        U, S, V = self._svd(filter)
        self.U = U
        self.S = S
        self.V = V

        self.sq_filter = filter**2
        U2, S2, V2 = self._svd(self.sq_filter)
        self.U2 = U2
        self.S2 = S2
        self.V2 = V2

    def mm(self, z):
        ''' Right multiply z by this MCC matrix. '''
        z_as_img = z.reshape([self.macro_shape[1]] + self.block_shape)
        h_as_img = self.U(self._scale(self.V.T(z_as_img)))
        h = h_as_img.reshape([-1, 1])
        return h

    def sq_mm(self, z):
        ''' Right multiply z by this MCC matrix. '''
        z_as_img = z.reshape([self.macro_shape[1]] + self.block_shape)
        h_as_img = self.U2(self._scale(self.V2.T(z_as_img), sq=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def T_mm(self, z):
        z_as_img = z.reshape([self.macro_shape[0]] + self.block_shape)
        h_as_img = self.V(self._scale(self.U.T(z_as_img), transpose=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def T_sq_mm(self, z):
        z_as_img = z.reshape([self.macro_shape[0]] + self.block_shape)
        h_as_img = self.V2(self._scale(self.U2.T(z_as_img), sq=True, transpose=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def _scale(self, z, sq=False, transpose=False):
        '''
        Apply diagonal singular values to input z. (ie. compute S z).
        Note: _scale is not using a VirtualMCCMatrix because that would require densifying S
        '''
        m, n = self.macro_shape
        if(transpose):
            m, n = n, m

        if(m < n): # input dim < output dim, ignore some dimensions
            z = z[ (slice(None),)*self.block_order + (slice(0,m),) ]

        if(sq):
            if(transpose):
                z = xp.conj(self.S2) * z
            else:
                z = self.S2 * z
        else:
            if(transpose):
                z = xp.conj(self.S) * z
            else:
                z = self.S * z

        if(m > n): # output dim > input dim, add some zero padding
            z = xp.pad(z, [(0, 0)]*self.block_order + [(0, m-n)])

        return z

    def _svd(self, conv_filter):
        '''
        Compute quantities for a sparse svd of the MCC matrix.
        '''
        macro_indices = [0, 1]
        block_indices = [2+r for r in range(self.block_order)]
        m, n = self.macro_shape

        conv_fft = xp.flip(conv_filter, axis=tuple(block_indices))

        zeropad = [(0, 0), (0, 0)] + [ (0, self.block_shape[i] - self.filter_shape[i]) for i in range(self.block_order)]
        roll = [-(self.filter_shape[i] - 1)//2 for i in range(self.block_order)]

        conv_fft = xp.pad(conv_fft, zeropad)
        conv_fft = xp.roll(conv_fft, roll, axis=block_indices)

        # Note: the SVD of a convolution matrix is F @ diag(F'^T c) @ F^T
        # where F^T is the unitary DFT matrix, and F'^T is the non-unitary DFT matrix
        conv_fft = xp.fft.fftn(conv_fft, axes=block_indices).transpose(block_indices + macro_indices)

        U, S, Vt = xp.linalg.svd(conv_fft)
        V = xp.conj(xp.swapaxes(Vt, -2, -1))
        return _VirtualMCCFactor(U), S, _VirtualMCCFactor(V)

    def densify(self):
        U = self.U.densify()
        Vt = xp.conj(self.V.densify()).T

        m, n = self.macro_shape

        if (m < n):  # input dim < output dim, ignore some dimensions
            Vt = Vt.reshape((np.prod(self.block_shape), n, n * np.prod(self.block_shape)))
            Vt = Vt[:, 0:m, :].reshape(np.prod(self.block_shape) * m, np.prod(self.block_shape) * n)

        SVt = self.S.reshape((-1, 1)) * Vt
        return xp.real(U @ SVt[0:U.shape[1], :]) # account for m > n case.

class MultiConvLayer(LinearLayer):
    def __init__(self, filter, block_shape, noise_std=0.001):
        super(MultiConvLayer, self).__init__(filter, noise_std=noise_std)
        m, n = filter.shape[:2]
        filter_shape = filter.shape[2:]

        self.macro_shape = (m, n)
        self.block_shape = list(block_shape)
        self.block_order = len(block_shape)
        self.filter_shape = filter_shape

        self.shape = (m * np.prod(block_shape), n * np.prod(block_shape))

        if(self.block_order > 24):
            raise ValueError(f"Input data blocks have tensor order {self.block_order} > 24 which will \
            break einstein sums used in this implementation.")

        self.filter = filter
        U, S, V = self._svd(filter)
        self.U = U
        self.S = S
        self.V = V

        self.sq_filter = filter**2
        U2, S2, V2 = self._svd(self.sq_filter)
        self.U2 = U2
        self.S2 = S2
        self.V2 = V2

    def mm(self, z):
        ''' Right multiply z by this MCC matrix. '''
        z_as_img = z.reshape([self.macro_shape[1]] + self.block_shape)
        h_as_img = self.U(self._scale(self.V.T(z_as_img)))
        h = h_as_img.reshape([-1, 1])
        return h

    def sq_mm(self, z):
        ''' Right multiply z by this MCC matrix. '''
        z_as_img = z.reshape([self.macro_shape[1]] + self.block_shape)
        h_as_img = self.U2(self._scale(self.V2.T(z_as_img), sq=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def T_mm(self, z):
        z_as_img = z.reshape([self.macro_shape[0]] + self.block_shape)
        h_as_img = self.V(self._scale(self.U.T(z_as_img), transpose=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def T_sq_mm(self, z):
        z_as_img = z.reshape([self.macro_shape[0]] + self.block_shape)
        h_as_img = self.V2(self._scale(self.U2.T(z_as_img), sq=True, transpose=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def apply(self, z):
        return self.mm(z) + xp.random.normal(size=(self.shape[0], 1), scale=self.noise_std)

    def _scale(self, z, sq=False, transpose=False):
        '''
        Apply diagonal singular values to input z. (ie. compute S z).
        Note: _scale is not using a VirtualMCCMatrix because that would require densifying S
        '''
        m, n = self.macro_shape
        if(transpose):
            m, n = n, m

        if(m < n): # input dim < output dim, ignore some dimensions
            z = z[ (slice(None),)*self.block_order + (slice(0,m),) ]

        if(sq):
            if(transpose):
                z = xp.conj(self.S2) * z
            else:
                z = self.S2 * z
        else:
            if(transpose):
                z = xp.conj(self.S) * z
            else:
                z = self.S * z

        if(m > n): # output dim > input dim, add some zero padding
            z = xp.pad(z, [(0, 0)]*self.block_order + [(0, m-n)])

        return z

    def _svd(self, conv_filter):
        '''
        Compute quantities for a sparse svd of the MCC matrix.
        '''
        macro_indices = [0, 1]
        block_indices = [2+r for r in range(self.block_order)]
        m, n = self.macro_shape

        conv_fft = xp.flip(conv_filter, axis=tuple(block_indices))

        zeropad = [(0, 0), (0, 0)] + [ (0, self.block_shape[i] - self.filter_shape[i]) for i in range(self.block_order)]
        roll = [-(self.filter_shape[i] - 1)//2 for i in range(self.block_order)]

        conv_fft = xp.pad(conv_fft, zeropad)
        conv_fft = xp.roll(conv_fft, roll, axis=block_indices)

        # Note: the SVD of a convolution matrix is F @ diag(F'^T c) @ F^T
        # where F^T is the unitary DFT matrix, and F'^T is the non-unitary DFT matrix
        conv_fft = xp.fft.fftn(conv_fft, axes=block_indices).transpose(block_indices + macro_indices)

        U, S, Vt = xp.linalg.svd(conv_fft)
        V = xp.conj(xp.swapaxes(Vt, -2, -1))
        return _VirtualMCCFactor(U), S, _VirtualMCCFactor(V)

    def densify(self):
        U = self.U.densify()
        Vt = xp.conj(self.V.densify()).T

        m, n = self.macro_shape

        if (m < n):  # input dim < output dim, ignore some dimensions
            Vt = Vt.reshape((np.prod(self.block_shape), n, n * np.prod(self.block_shape)))
            Vt = Vt[:, 0:m, :].reshape(np.prod(self.block_shape) * m, np.prod(self.block_shape) * n)

        SVt = self.S.reshape((-1, 1)) * Vt
        return xp.real(U @ SVt[0:U.shape[1], :]) # account for m > n case.

class MultiConvChannel(LinearChannel):
    def __init__(self, measurements, filter, block_shape, noise_std=0.001):
        super(MultiConvChannel, self).__init__(measurements, filter, noise_std=noise_std)
        m, n = filter.shape[:2]
        filter_shape = filter.shape[2:]

        self.macro_shape = (m, n)
        self.block_shape = list(block_shape)
        self.block_order = len(block_shape)
        self.filter_shape = filter_shape

        self.shape = (m * np.prod(block_shape), n * np.prod(block_shape))

        if(self.block_order > 24):
            raise ValueError(f"Input data blocks have tensor order {self.block_order} > 24 which will \
            break einstein sums used in this implementation.")

        self.filter = filter
        U, S, V = self._svd(filter)
        self.U = U
        self.S = S
        self.V = V

        self.sq_filter = filter**2
        U2, S2, V2 = self._svd(self.sq_filter)
        self.U2 = U2
        self.S2 = S2
        self.V2 = V2

    def mm(self, z):
        z_as_img = z.reshape([self.macro_shape[1]] + self.block_shape)
        h_as_img = self.U(self._scale(self.V.T(z_as_img)))
        h = h_as_img.reshape([-1, 1])
        return h

    def sq_mm(self, z):
        z_as_img = z.reshape([self.macro_shape[1]] + self.block_shape)
        h_as_img = self.U2(self._scale(self.V2.T(z_as_img), sq=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def T_mm(self, z):
        z_as_img = z.reshape([self.macro_shape[0]] + self.block_shape)
        h_as_img = self.V(self._scale(self.U.T(z_as_img), transpose=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def T_sq_mm(self, z):
        z_as_img = z.reshape([self.macro_shape[0]] + self.block_shape)
        h_as_img = self.V2(self._scale(self.U2.T(z_as_img), sq=True, transpose=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def apply(self, z):
        return self.mm(z) + xp.random.normal(size=(self.shape[0], 1), scale=self.noise_std)

    def _scale(self, z, sq=False, transpose=False):
        '''
        Apply diagonal singular values to input z. (ie. compute S z).
        Note: _scale is not using a VirtualMCCMatrix because that would require densifying S
        '''
        m, n = self.macro_shape
        if(transpose):
            m, n = n, m

        if(m < n): # input dim < output dim, ignore some dimensions
            z = z[ (slice(None),)*self.block_order + (slice(0,m),) ]

        if(sq):
            if(transpose):
                z = xp.conj(self.S2) * z
            else:
                z = self.S2 * z
        else:
            if(transpose):
                z = xp.conj(self.S) * z
            else:
                z = self.S * z

        if(m > n): # output dim > input dim, add some zero padding
            z = xp.pad(z, [(0, 0)]*self.block_order + [(0, m-n)])

        return z

    def _svd(self, conv_filter):
        '''
        Compute quantities for a sparse svd of the MCC matrix.
        '''
        macro_indices = [0, 1]
        block_indices = [2+r for r in range(self.block_order)]
        m, n = self.macro_shape

        conv_fft = xp.flip(conv_filter, axis=tuple(block_indices))

        zeropad = [(0, 0), (0, 0)] + [ (0, self.block_shape[i] - self.filter_shape[i]) for i in range(self.block_order)]
        roll = [-(self.filter_shape[i] - 1)//2 for i in range(self.block_order)]

        conv_fft = xp.pad(conv_fft, zeropad)
        conv_fft = xp.roll(conv_fft, roll, axis=block_indices)

        # Note: the SVD of a convolution matrix is F @ diag(F'^T c) @ F^T
        # where F^T is the unitary DFT matrix, and F'^T is the non-unitary DFT matrix
        conv_fft = xp.fft.fftn(conv_fft, axes=block_indices).transpose(block_indices + macro_indices)

        U, S, Vt = xp.linalg.svd(conv_fft)
        V = xp.conj(xp.swapaxes(Vt, -2, -1))
        return _VirtualMCCFactor(U), S, _VirtualMCCFactor(V)

    def densify(self):
        U = self.U.densify()
        Vt = xp.conj(self.V.densify()).T

        m, n = self.macro_shape

        if (m < n):  # input dim < output dim, ignore some dimensions
            Vt = Vt.reshape((np.prod(self.block_shape), n, n * np.prod(self.block_shape)))
            Vt = Vt[:, 0:m, :].reshape(np.prod(self.block_shape) * m, np.prod(self.block_shape) * n)

        SVt = self.S.reshape((-1, 1)) * Vt
        return xp.real(U @ SVt[0:U.shape[1], :]) # account for m > n case.

class _VirtualMCCFactor:
    def __init__(self, sparse_mat, with_fft=True):
        '''
        This nested class behaves like a numpy matrix and wraps sparse MCC matrices resulting from SVD.
        Parameters
        ----------
        sparse_mat: sparse MCC factor tensor of the form (d1, ..., dk, m, n)
        with_fft: if True, __matmul__ applies a block inverse fourier transform after multiplication by sparse_mat
            and __tmatmul__ applies the corresponding block fourier transform before multiplication by sparse_mat^T.
        '''
        self.with_fft = with_fft
        self.sparse_mat = sparse_mat
        self.block_shape = sparse_mat.shape[:-2]
        self.block_order = len(self.block_shape)
        self.macro_shape = sparse_mat.shape[-2:]

    def __tmatmul__(self, z):
        # transpose matrix multiplication
        if(self.with_fft):
            block_indices = [1 + r for r in range(self.block_order)]
            z = xp.transpose(xp.fft.fftn(z, axes=block_indices, norm="ortho"), block_indices + [0])

        idxs = "abcdefghijklmnopqrstuvw"[:self.block_order]
        return xp.einsum(f'{idxs}yz,{idxs}y->{idxs}z', xp.conj(self.sparse_mat), z) # implicit transpose

    def __matmul__(self, z):
        idxs = "abcdefghijklmnopqrstuvw"[:self.block_order]
        z = xp.einsum(f'{idxs}yz,{idxs}z->{idxs}y', self.sparse_mat, z)

        if(self.with_fft):
            block_indices = list(range(self.block_order))
            macro_idx = 1 + block_indices[-1]
            z = xp.real(xp.fft.ifftn(xp.transpose(z, [macro_idx] + block_indices), axes=[1+r for r in block_indices], norm="ortho"))
        return z

    def __call__(self, z):
        return self @ z

    def T(self, z):
        return self.__tmatmul__(z)

    def densify(self):
        # Technically, this method densifies the transpose and returns a double transpose
        dim = np.prod(self.block_shape) * self.macro_shape[-1]
        natural_basis = xp.eye(dim).reshape( (self.macro_shape[-1],) + self.block_shape + (-1,) )

        if(self.with_fft):
            block_indices = [1 + r for r in range(self.block_order)]
            natural_basis = xp.transpose(xp.fft.fftn(natural_basis, axes=block_indices, norm="ortho"), block_indices + [0, -1])

        idxs = "abcdefghijklmnopqrstuvw"[:self.block_order]
        mat = xp.einsum(f'{idxs}yz,{idxs}yx->{idxs}zx', xp.conj(self.sparse_mat), natural_basis)
        return xp.conj(mat.reshape((-1, dim))).T

class AbsMultiConvLayer(AbsLinearLayer):
    def __init__(self, filter, block_shape):
        super(AbsMultiConvLayer, self).__init__(filter)
        m, n = filter.shape[:2]
        filter_shape = filter.shape[2:]

        self.macro_shape = (m, n)
        self.block_shape = list(block_shape)
        self.block_order = len(block_shape)
        self.filter_shape = filter_shape

        self.shape = (m * np.prod(block_shape), n * np.prod(block_shape))

        if (self.block_order > 24):
            raise ValueError(f"Input data blocks have tensor order {self.block_order} > 24 which will \
            break einstein sums used in this implementation.")

        self.filter = filter
        U, S, V = self._svd(filter)
        self.U = U
        self.S = S
        self.V = V

        self.sq_filter = filter ** 2
        U2, S2, V2 = self._svd(self.sq_filter)
        self.U2 = U2
        self.S2 = S2
        self.V2 = V2

    def mm(self, z):
        ''' Right multiply z by this MCC matrix. '''
        z_as_img = z.reshape([self.macro_shape[1]] + self.block_shape)
        h_as_img = self.U(self._scale(self.V.T(z_as_img)))
        h = h_as_img.reshape([-1, 1])
        return h

    def sq_mm(self, z):
        ''' Right multiply z by this MCC matrix. '''
        z_as_img = z.reshape([self.macro_shape[1]] + self.block_shape)
        h_as_img = self.U2(self._scale(self.V2.T(z_as_img), sq=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def T_mm(self, z):
        z_as_img = z.reshape([self.macro_shape[0]] + self.block_shape)
        h_as_img = self.V(self._scale(self.U.T(z_as_img), transpose=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def T_sq_mm(self, z):
        z_as_img = z.reshape([self.macro_shape[0]] + self.block_shape)
        h_as_img = self.V2(self._scale(self.U2.T(z_as_img), sq=True, transpose=True))
        h = h_as_img.reshape([-1, 1])
        return h

    def apply(self, z):
        return self.mm(z) + xp.random.normal(size=(self.shape[0], 1), scale=self.noise_std)

    def _scale(self, z, sq=False, transpose=False):
        '''
        Apply diagonal singular values to input z. (ie. compute S z).
        Note: _scale is not using a VirtualMCCMatrix because that would require densifying S
        '''
        m, n = self.macro_shape
        if (transpose):
            m, n = n, m

        if (m < n):  # input dim < output dim, ignore some dimensions
            z = z[(slice(None),) * self.block_order + (slice(0, m),)]

        if (sq):
            if (transpose):
                z = xp.conj(self.S2) * z
            else:
                z = self.S2 * z
        else:
            if (transpose):
                z = xp.conj(self.S) * z
            else:
                z = self.S * z

        if (m > n):  # output dim > input dim, add some zero padding
            z = xp.pad(z, [(0, 0)] * self.block_order + [(0, m - n)])

        return z

    def _svd(self, conv_filter):
        '''
        Compute quantities for a sparse svd of the MCC matrix.
        '''
        macro_indices = [0, 1]
        block_indices = [2 + r for r in range(self.block_order)]
        m, n = self.macro_shape

        conv_fft = xp.flip(conv_filter, axis=tuple(block_indices))

        zeropad = [(0, 0), (0, 0)] + [(0, self.block_shape[i] - self.filter_shape[i]) for i in
                                      range(self.block_order)]
        roll = [-(self.filter_shape[i] - 1) // 2 for i in range(self.block_order)]

        conv_fft = xp.pad(conv_fft, zeropad)
        conv_fft = xp.roll(conv_fft, roll, axis=block_indices)

        # Note: the SVD of a convolution matrix is F @ diag(F'^T c) @ F^T
        # where F^T is the unitary DFT matrix, and F'^T is the non-unitary DFT matrix
        conv_fft = xp.fft.fftn(conv_fft, axes=block_indices).transpose(block_indices + macro_indices)

        U, S, Vt = xp.linalg.svd(conv_fft)
        V = xp.conj(xp.swapaxes(Vt, -2, -1))
        return _VirtualMCCFactor(U), S, _VirtualMCCFactor(V)

    def densify(self):
        U = self.U.densify()
        Vt = xp.conj(self.V.densify()).T

        m, n = self.macro_shape

        if (m < n):  # input dim < output dim, ignore some dimensions
            Vt = Vt.reshape((np.prod(self.block_shape), n, n * np.prod(self.block_shape)))
            Vt = Vt[:, 0:m, :].reshape(np.prod(self.block_shape) * m, np.prod(self.block_shape) * n)

        SVt = self.S.reshape((-1, 1)) * Vt
        return xp.real(U @ SVt[0:U.shape[1], :])  # account for m > n case.


class ReLUMultiConvLayer(PiecewiseLinearMultiConvLayer):
    def __init__(self, filter, block_shape, monte_carlo=False):
        lines = [{"zmin": -xp.inf, "zmax": 0, "x0": 0, "slope": 0}, {"zmin": 0, "zmax": xp.inf, "x0": 0, "slope": 1}]
        super(ReLUMultiConvLayer, self).__init__(filter=filter, block_shape=block_shape, lines=lines,
                                                 monte_carlo=monte_carlo)

class SparseMultiConvChannel(LinearChannel):
    def __init__(self, measurements, filter, block_shape, noise_std=0.001):
        super(SparseMultiConvChannel, self).__init__(measurements, filter, noise_std=noise_std)
        assert(len(block_shape) == 1, "This channel only supports 1-D signals")
        m, n = filter.shape[:2]
        q = block_shape[0]
        filter_shape = filter.shape[2:]

        self.macro_shape = (m, n)
        self.block_len = q
        self.filter_shape = filter_shape

        self.shape = (m * np.prod(block_shape), n * np.prod(block_shape))

        self.filter = filter
        self.sq_filter = filter**2
    
    def mm(self, z):
        m, n = self.macro_shape
        q = self.block_len
        z_ = z.reshape(n, q)
        result = fast_1d_multiconv(z_, self.filter, q)
        return result.reshape((-1, 1))

    def sq_mm(self, z):
        m, n = self.macro_shape
        q = self.block_len
        z_ = z.reshape(n, q)
        result = fast_1d_multiconv(z_, self.sq_filter, q)
        return result.reshape((-1, 1))

    def T_mm(self, z):
        m, n = self.macro_shape
        q = self.block_len
        z_ = z.reshape(m, q)
        result = fast_1d_T_multiconv(z_, self.filter, q)
        return result.reshape((-1, 1))

    def T_sq_mm(self, z):
        m, n = self.macro_shape
        q = self.block_len
        z_ = z.reshape(m, q)
        result = fast_1d_T_multiconv(z_, self.sq_filter, q)
        return result.reshape((-1, 1))

    def apply(self, z):
        return self.mm(z) + xp.random.normal(size=(self.shape[0], 1), scale=self.noise_std)

class SparseMultiConvLayer(LinearLayer):
    def __init__(self, filter, block_shape, noise_std=0.001):
        super(SparseMultiConvLayer, self).__init__(filter, noise_std=noise_std)
        m, n = filter.shape[:2]
        q = block_shape[0]
        filter_shape = filter.shape[2:]

        self.macro_shape = (m, n)
        self.block_len = q
        self.filter_shape = filter_shape

        self.shape = (m * np.prod(block_shape), n * np.prod(block_shape))

        self.filter = filter
        self.sq_filter = filter**2


    def mm(self, z):
        m, n = self.macro_shape
        q = self.block_len
        z_ = z.reshape(n, q)
        result = fast_1d_multiconv(z_, self.filter, q)
        return result.reshape((-1, 1))

    def sq_mm(self, z):
        m, n = self.macro_shape
        q = self.block_len
        z_ = z.reshape(n, q)
        result = fast_1d_multiconv(z_, self.sq_filter, q)
        return result.reshape((-1, 1))

    def T_mm(self, z):
        m, n = self.macro_shape
        q = self.block_len
        z_ = z.reshape(m, q)
        result = fast_1d_T_multiconv(z_, self.filter, q)
        return result.reshape((-1, 1))

    def T_sq_mm(self, z):
        m, n = self.macro_shape
        q = self.block_len
        z_ = z.reshape(m, q)
        result = fast_1d_T_multiconv(z_, self.sq_filter, q)
        return result.reshape((-1, 1))

    def apply(self, z):
        return self.mm(z) + xp.random.normal(size=(self.shape[0], 1), scale=self.noise_std)
