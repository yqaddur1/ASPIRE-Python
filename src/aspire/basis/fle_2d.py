import logging

import numpy as np
import scipy.sparse as sparse
from scipy.fft import dct, idct
from scipy.special import jv

from aspire.basis import FBBasisMixin, SteerableBasis2D
from aspire.basis.basis_utils import besselj_zeros
from aspire.basis.fle_2d_utils import (
    barycentric_interp_sparse,
    fle_ell_sign,
    precomp_transform_complex_to_real,
    transform_complex_to_real,
)
from aspire.nufft import anufft, nufft
from aspire.numeric import fft
from aspire.utils import grid_2d

logger = logging.getLogger(__name__)


class FLEBasis2D(SteerableBasis2D, FBBasisMixin):
    """
    Define a derived class for Fast Fourier Bessel expansion for 2D images
        using interpolation from Chebyshev nodes.
    The algorithms used are described in the following publication:
    N. F. Marshall, O. Mickelin, A. Singer, Fast Expansion into Harmonics on the Disk:
        A Steerable Basis with Fast Radial Convolution. (submitted)

    https://arxiv.org/pdf/2207.13674.pdf
    """

    def __init__(
        self, size, bandlimit=None, epsilon=1e-10, match_fb=False, dtype=np.float32
    ):
        """
        :param size: The size of the vectors for which to define the FLE basis.
                 Currently only square images are supported.
        :param bandlimit: Maximum frequency band for computing basis functions. Note that the
            `ell_max` of other Basis objects is computed *from* the bandlimit for the FLE basis.
             Defaults to the resolution of the basis.
        :param epsilon: Relative precision between FLE fast method and dense matrix multiplication.
        :param match_fb: If this flag is set, the number of basis functions will be forced to match
            `FBBasis2D` for the same resolution. In particular, the thresholding process will not
            be performed, meaning `self.count` will be larger.
        :param dtype: Datatype of images and coefficients represented.
        """
        if isinstance(size, int):
            size = (size, size)
        ndim = len(size)
        assert ndim == 2, "Only two-dimensional basis functions are supported."
        assert len(set(size)) == 1, "Only square domains are supported"

        self.bandlimit = bandlimit
        self.epsilon = epsilon
        self.match_fb = match_fb
        self.dtype = dtype
        super().__init__(size, ell_max=None, dtype=self.dtype)

    def _build(self):
        """
        Build the internal data structure for the FLEBasis2D class.
        """

        # bandlimit set to basis size by default
        if not self.bandlimit:
            self.bandlimit = self.nres

        # compute number of k's for each ell
        self._calc_k_max()

        if self.match_fb:
            # Use FB2D and FFB2D heuristic for computing max basis functions
            self.max_basis_functions = self.k_max[0] + sum(2 * self.k_max[1:])
        else:
            # Regular Fourier-Bessel bandlimit (equivalent to pi*R**2)
            # Final self.count will be < self.max_basis_functions
            # See self._threshold_basis_functions()
            self.max_basis_functions = int(self.nres**2 * np.pi / 4)

        self._compute_maxitr_and_numsparse()

        self._compute_cartesian_gridpoints()

        self._precomp()

        # Steerable basis indices
        self._build_indices()

        # FB compatability indices
        self._get_fb_compat_indices()

    def _build_indices(self):
        self.angular_indices = np.abs(self.ells)
        self.radial_indices = self.ks - 1
        self.signs_indices = np.array(list(map(fle_ell_sign, self.ells)))

    def indices(self):
        """
        Return the precomputed indices for each basis function.
        """
        return {
            "ells": self.angular_indices,
            "ks": self.radial_indices,
            "sgns": self.signs_indices,
        }

    def _get_fb_compat_indices(self):
        ind = self.indices()
        self.fb_compat_indices = np.lexsort((ind["ks"], ind["sgns"], ind["ells"]))
        self.fb_compat_indices_t = np.zeros(self.count, dtype=int)

    def _precomp(self):
        """
        Precompute the basis functions and other objects used in the evaluation of
            coefficients.
        """

        # Find bessel functions zeros (the eigenvalues of the Laplacian on
        # the disk) and generate the FLE Basis functions
        self._lap_eig_disk()

        # Some important constants
        self.smallest_lambda = np.min(self.bessel_zeros)
        self.greatest_lambda = np.max(self.bessel_zeros)
        self.max_ell = np.max(np.abs(self.ells))
        self.h = 1 / (self.nres // 2)

        # give each ell a positive index increasing first in |ell|
        # then in sign, e.g. 0->1, -1->2, 1->3, -2->4, 2->5, etc.
        self.ells_p = 2 * np.abs(self.ells) - (self.ells < 0)
        self.ell_p_max = np.max(self.ells_p)
        # idx_list[k] contains the indices j of ells_p where ells_p[j] = k
        idx_list = [[] for i in range(self.ell_p_max + 1)]
        for i in range(self.count):
            ellp = self.ells_p[i]
            idx_list[ellp].append(i)
        self.idx_list = idx_list

        # real <-> complex
        self.c2r = precomp_transform_complex_to_real(self.ells)
        self.r2c = sparse.csr_matrix(self.c2r.transpose().conj())

        # create self.nus to be able to access the physical value of ell,
        # rather than the reindexed version
        self.nus = np.zeros(1 + 2 * self.max_ell, dtype=int)
        self.nus[0] = 0
        for i in range(1, self.max_ell + 1):
            self.nus[2 * i - 1] = -i
            self.nus[2 * i] = i
        self.c2r_nus = precomp_transform_complex_to_real(self.nus)
        self.r2c_nus = sparse.csr_matrix(self.c2r_nus.transpose().conj())

        # radial and angular nodes for NUFFT
        self._compute_nufft_points()
        self.num_interp = self.num_radial_nodes
        if self.numsparse > 0:
            self.num_interp = 2 * self.num_radial_nodes

        self._build_interpolation_matrix()

    def _compute_maxitr_and_numsparse(self):
        """
        Uses heuristics from paper to assign self.maxitr and self.numsparse.
        """
        # maxitr: maximum number of iterations for numerically solving linear
        # system in self.evaluate()
        maxitr = 1 + int(3 * np.log2(self.nres))
        # numsparse: parameter used to create sparse Chebyshev interpolation matrix
        # see self._build_interpolation_matrix()
        numsparse = 32
        if self.epsilon >= 1e-10:
            numsparse = 22
            maxitr = 1 + int(2 * np.log2(self.nres))
        if self.epsilon >= 1e-7:
            numsparse = 16
            maxitr = 1 + int(np.log2(self.nres))
        if self.epsilon >= 1e-4:
            numsparse = 8
            maxitr = 1 + int(np.log2(self.nres)) // 2
        self.maxitr = maxitr
        self.numsparse = numsparse

    def _compute_cartesian_gridpoints(self):
        """
        Creates meshgrids based on basis size.
        """
        if self.match_fb:
            # creates correct odd-resolution grid
            # matching other FB classes
            grid = grid_2d(self.nres)
            self.xs = grid["x"]
            self.ys = grid["y"]
            self.rs = grid["r"]
        else:
            # original implementation
            R = self.nres // 2
            x = np.arange(-R, R + self.nres % 2)
            y = np.arange(-R, R + self.nres % 2)
            xs, ys = np.meshgrid(x, y)
            self.xs, self.ys = xs / R, ys / R
            self.rs = np.sqrt(self.xs**2 + self.ys**2)
        self.radial_mask = self.rs > 1 + 1e-13

    def _compute_nufft_points(self):
        """
        Computes gridpoints for the non-uniform FFT.
        """

        # Number of radial nodes
        # (Lemma 4.1)
        # compute max {2.4 * self.nres , Log2 ( 1 / epsilon) }
        Q = int(np.ceil(2.4 * self.nres))
        num_radial_nodes = Q
        tmp = 1 / (np.sqrt(np.pi))
        for q in range(1, Q + 1):
            tmp = tmp / q * (np.sqrt(np.pi) * self.nres / 4)
            if tmp <= self.epsilon:
                num_radial_nodes = int(max(q, np.log2(1 / self.epsilon)))
                break
        self.num_radial_nodes = max(
            num_radial_nodes, int(np.ceil(np.log2(1 / self.epsilon)))
        )

        # Number of angular nodes
        # (Lemma 4.2)
        # compute max {7.08 * self.nres, Log2(1/epsilon) + Log2(self.nres**2) }

        S = int(max(7.08 * self.nres, -np.log2(self.epsilon) + 2 * np.log2(self.nres)))
        num_angular_nodes = S
        for s in range(int(self.greatest_lambda + self.ell_p_max) + 1, S + 1):
            tmp = self.nres**2 * ((self.greatest_lambda + self.ell_p_max) / s) ** s
            if tmp <= self.epsilon:
                num_angular_nodes = int(max(int(s), np.log2(1 / self.epsilon)))
                break

        # must be even
        if num_angular_nodes % 2 == 1:
            num_angular_nodes += 1

        self.num_angular_nodes = num_angular_nodes

        # create gridpoints
        nodes = 1 - (2 * np.arange(self.num_radial_nodes) + 1) / (
            2 * self.num_radial_nodes
        )
        nodes = (np.cos(np.pi * nodes) + 1) / 2
        nodes = (
            self.greatest_lambda - self.smallest_lambda
        ) * nodes + self.smallest_lambda
        nodes = nodes.reshape(self.num_radial_nodes, 1)

        radius = self.nres // 2
        h = 1 / radius

        phi = (
            2 * np.pi * np.arange(self.num_angular_nodes // 2) / self.num_angular_nodes
        )
        x = np.cos(phi).reshape(1, self.num_angular_nodes // 2)
        y = np.sin(phi).reshape(1, self.num_angular_nodes // 2)
        x = x * nodes * h
        y = y * nodes * h
        self.grid_x = x.flatten()
        self.grid_y = y.flatten()

    def _build_interpolation_matrix(self):
        """
        Create the matrix used in the third step of evaluate_t() and the first step of evaluate()
        for barycentric interpolation from Chebyshev nodes.
        """
        A3 = [None] * (self.ell_p_max + 1)
        A3_T = [None] * (self.ell_p_max + 1)
        # known points from which to interpolate Beta values to desired points
        known_points = np.cos(
            np.pi * (1 - (2 * np.arange(self.num_interp) + 1) / (2 * self.num_interp))
        )
        for i in range(self.ell_p_max + 1):
            # target points to evaluate Betas
            target_points = (
                2
                * (self.bessel_zeros[self.idx_list[i]] - self.smallest_lambda)
                / (self.greatest_lambda - self.smallest_lambda)
                - 1
            )

            A3[i], A3_T[i] = barycentric_interp_sparse(
                target_points, known_points, self.numsparse
            )
        self.A3 = A3
        self.A3_T = A3_T

    def _lap_eig_disk(self):
        """
        Compute the eigenvalues of the Laplacian operator on a disk with Dirichlet boundary conditions.
        """
        # max number of Bessel function orders being considered
        max_ell = int(3 * np.sqrt(self.max_basis_functions))
        # max number of zeros per Bessel function (number of frequencies per bessel)
        max_k = int(2 * np.sqrt(self.max_basis_functions))

        # preallocate containers for roots
        # 0 frequency plus pos and negative frequencies for each bessel function
        # num functions per frequency
        num_ells = 1 + 2 * max_ell
        self.ells = np.zeros((num_ells, max_k), dtype=int)
        self.ks = np.zeros((num_ells, max_k), dtype=int)
        self.bessel_zeros = np.ones((num_ells, max_k), dtype=np.float64) * np.Inf

        # keep track of which order Bessel function we're on
        self.ells[0, :] = 0
        # bessel_roots[0, m] is the m'th zero of J_0
        self.bessel_zeros[0, :] = besselj_zeros(0, max_k)
        # table of values of which zero of J_0 we are finding
        self.ks[0, :] = np.arange(max_k) + 1

        # add roots of J_ell for ell>0 twice with +k and -k (frequencies)
        # iterate over Bessel function order
        for ell in range(1, max_ell + 1):
            self.ells[2 * ell - 1, :] = -ell
            self.ks[2 * ell - 1, :] = np.arange(max_k) + 1

            self.bessel_zeros[2 * ell - 1, :max_k] = besselj_zeros(ell, max_k)

            self.ells[2 * ell, :] = ell
            self.ks[2 * ell, :] = self.ks[2 * ell - 1, :]
            self.bessel_zeros[2 * ell, :] = self.bessel_zeros[2 * ell - 1, :]

        # Reshape the arrays and order by the size of the Bessel function zeros
        self._flatten_and_sort_bessel_zeros()

        # Apply threshold criterion to throw out some basis functions
        # Grab final number of basis functions for this Basis
        self.count = self._threshold_basis_functions()

        self._create_basis_functions()

    def _flatten_and_sort_bessel_zeros(self):
        """
        Reshapes arrays self.ells, self.ks, and self.bessel_zeros
        """
        # flatten list of zeros, ells and ks:
        self.ells = self.ells.flatten()
        self.ks = self.ks.flatten()
        self.bessel_zeros = self.bessel_zeros.flatten()

        idx = np.argsort(self.bessel_zeros)
        self.ells = self.ells[idx]
        self.ks = self.ks[idx]
        self.bessel_zeros = self.bessel_zeros[idx]

        # sort complex conjugate pairs: -ell first, +ell second
        idx = np.arange(self.max_basis_functions + 1)
        for i in range(self.max_basis_functions + 1):
            if self.ells[i] >= 0:
                continue
            if np.abs(self.bessel_zeros[i] - self.bessel_zeros[i + 1]) < 1e-14:
                continue
            idx[i - 1] = i
            idx[i] = i - 1

        self.ells = self.ells[idx]
        self.ks = self.ks[idx]
        self.bessel_zeros = self.bessel_zeros[idx]

    def _threshold_basis_functions(self):
        """
        Implements the bandlimit threshold which caps the number of basis functions
        that are actually required.
        :return: The final overall number of basis functions to be used.
        """
        # Maximum bandlimit
        # (Section 4.1)
        # Can remove frequencies above this threshold based on the fact that
        # there should not be more basis functions than pixels contained in the
        # unit disk inscribed on the image
        _final_num_basis_functions = self.max_basis_functions

        # implement FLE thresholding unless we want to match count of other FB bases
        if not self.match_fb:
            for _ in range(len(self.bessel_zeros)):
                if (
                    self.bessel_zeros[_final_num_basis_functions] / (np.pi)
                    >= (self.bandlimit - 1) // 2
                ):
                    _final_num_basis_functions -= 1

        # potentially subtract one to keep complex conjugate pairs
        if self.ells[_final_num_basis_functions - 1] < 0:
            _final_num_basis_functions -= 1

        # discard zeros above the threshold
        self.ells = self.ells[:_final_num_basis_functions]
        self.ks = self.ks[:_final_num_basis_functions]
        self.bessel_zeros = self.bessel_zeros[:_final_num_basis_functions]

        return _final_num_basis_functions

    def _create_basis_functions(self):
        """
        Generate the actual basis functions as Python lambda operators
        """
        norm_constants = np.zeros(self.count)
        basis_functions = [None] * self.count
        for i in range(self.count):
            # parameters defining the basis function: bessel order and which bessel root
            ell = self.ells[i]
            bessel_zero = self.bessel_zeros[i]

            # compute normalization constant
            # see Eq. 6
            c = 1 / np.sqrt(np.pi * jv(ell + 1, bessel_zero) ** 2)
            # create function
            # See Eq. 1
            if ell == 0:
                basis_functions[i] = (
                    lambda r, t, c=c, ell=ell, bessel_zero=bessel_zero: c
                    * jv(ell, bessel_zero * r)
                    * (r <= 1)
                )
            else:
                basis_functions[i] = (
                    lambda r, t, c=c, ell=ell, bessel_zero=bessel_zero: c
                    * jv(ell, bessel_zero * r)
                    * np.exp(1j * ell * t)
                    * (-1) ** np.abs(ell)
                    * (r <= 1)
                )

            norm_constants[i] = c

        self.norm_constants = norm_constants
        self.basis_functions = basis_functions

    def _expand(self, imgs):
        """
        Alternative to `Basis.expand()`, which computes a conjugate gradient solution using
            `_evaluate` and `_evaluate_t`. Computes FLE coefficients from a stack of
            images in Cartesian coordinates.
        :param imgs: An Image object containing square images of size `self.nres`.
        :return: A NumPy array of size `(num_images, self.count)` containing the FLE
            coefficients.
        """
        coeffs = self.evaluate_t(imgs)
        tmp = coeffs
        for _ in range(self.maxitr):
            tmp = tmp - self.evaluate_t(self.evaluate(tmp)) + coeffs
        return tmp

    def _evaluate(self, coeffs):
        """
        Evaluates FLE coefficients and return in standard 2D Cartesian coordinates.

        :param v: A coefficient vector (or an array of coefficient vectors) to
            be evaluated. The last dimension must be equal to `self.count`
        :return: An Image object containing the corresponding images.
        """
        if self.match_fb:
            # sign of basis functions with positive indices flipped relative to FB2d
            flip_sign_indices = np.where(self.indices()["sgns"] == 1)
            coeffs[flip_sign_indices] *= -1.0
            # reorder coefficients by FB2d ordering
            coeffs = coeffs[self.fb_compat_indices]

        # See Remark 3.3 and Section 3.4
        betas = self._step3(coeffs)
        z = self._step2(betas)
        im = self._step1(z)
        return im

    def _evaluate_t(self, imgs):
        """
        Evaluate 2D Cartesian image(s) and return the corresponding FLE coefficients.

        :param imgs: An Image object containing square images of size `self.nres`.
        :return: A NumPy array of size `(num_images, self.count)` containing the FLE
            coefficients.
        """
        # See Section 3.5
        imgs = imgs.copy()
        imgs[:, self.radial_mask] = 0
        z = self._step1_t(imgs)
        b = self._step2_t(z)
        coeffs = self._step3_t(b)
        if self.match_fb:
            flip_signs_indices = np.where(self.indices()["sgns"] == 1)
            coeffs[:, flip_signs_indices] *= -1.0
            coeffs = coeffs[:, self.fb_compat_indices]
        return coeffs

    def _step1_t(self, im):
        """
        Step 1 of the adjoint transformation (images to coefficients).
        Calculates the NUFFT of the image on gridpoints `self.grid_x` and `self.grid_y`.
        """
        im = im.reshape(-1, self.nres, self.nres).astype(np.complex128)
        num_img = im.shape[0]
        z = np.zeros(
            (num_img, self.num_radial_nodes, self.num_angular_nodes),
            dtype=np.complex128,
        )
        _z = (
            nufft(im, np.stack((self.grid_x, self.grid_y)), epsilon=self.epsilon)
            * self.h**2
        )
        _z = _z.reshape(num_img, self.num_radial_nodes, self.num_angular_nodes // 2)
        z[:, :, : self.num_angular_nodes // 2] = _z
        z[:, :, self.num_angular_nodes // 2 :] = np.conj(_z)
        return z

    def _step2_t(self, z):
        """
        Step 2 of the adjoint transformation (images to coefficients).
        Computes values of the analytic functions Beta_n at the Chebyshev nodes.
        See Lemma 2.2.
        """
        num_img = z.shape[0]
        # Compute FFT along angular nodes
        betas = fft.fft(z, axis=2) / self.num_angular_nodes
        betas = betas[:, :, self.nus]
        betas = np.conj(betas)
        betas = np.swapaxes(betas, 0, 2)
        betas = betas.reshape(-1, self.num_radial_nodes * num_img)
        betas = self.c2r_nus @ betas
        betas = betas.reshape(-1, self.num_radial_nodes, num_img)
        betas = np.real(np.swapaxes(betas, 0, 2))
        return betas

    def _step3_t(self, betas):
        """
        Step 3 of the adjoint transformation (images to coefficients).
        Uses barycenteric interpolation to compute the values of the Betas
        at the Bessel roots to arrive at the Fourier-Bessel coefficients.
        """
        num_img = betas.shape[0]
        if self.num_interp > self.num_radial_nodes:
            betas = dct(betas, axis=1, type=2) / (2 * self.num_radial_nodes)
            zeros = np.zeros(betas.shape)
            betas = np.concatenate((betas, zeros), axis=1)
            betas = idct(betas, axis=1, type=2) * 2 * betas.shape[1]
        betas = np.moveaxis(betas, 0, -1)

        coeffs = np.zeros((self.count, num_img), dtype=np.float64)
        for i in range(self.ell_p_max + 1):
            coeffs[self.idx_list[i]] = self.A3[i] @ betas[:, i, :]
        coeffs = coeffs.T

        return coeffs * self.norm_constants / self.h

    def _step3(self, coeffs):
        """
        Adjoint of _step3_t and Step 1 of the forward transformation (coefficients
            to images).
        Uses barycenteric interpolation in reverse to compute values of Betas
            at Chebyshev nodes, given an array of FLE coefficients.
        """
        coeffs = coeffs.copy().reshape(-1, self.count)
        num_img = coeffs.shape[0]
        coeffs *= self.h * self.norm_constants
        coeffs = coeffs.T

        out = np.zeros(
            (self.num_interp, 2 * self.max_ell + 1, num_img),
            dtype=np.float64,
            order="F",
        )
        for i in range(self.ell_p_max + 1):
            out[:, i, :] = self.A3_T[i] @ coeffs[self.idx_list[i]]
        out = np.moveaxis(out, -1, 0)
        if self.num_interp > self.num_radial_nodes:
            out = dct(out, axis=1, type=2)
            out = out[:, : self.num_radial_nodes, :]
            out = idct(out, axis=1, type=2)

        return out

    def _step2(self, betas):
        """
        Adjoint of _step2_t and Step 2 of the forward transformation (coefficients
            to images).
        Uses the IFFT to convert Beta values into Fourier-space images.
        """
        num_img = betas.shape[0]
        tmp = np.zeros(
            (num_img, self.num_radial_nodes, self.num_angular_nodes),
            dtype=np.complex128,
        )

        betas = np.swapaxes(betas, 0, 2)
        betas = betas.reshape(-1, self.num_radial_nodes * num_img)
        betas = self.r2c_nus @ betas
        betas = betas.reshape(-1, self.num_radial_nodes, num_img)
        betas = np.swapaxes(betas, 0, 2)

        tmp[:, :, self.nus] = np.conj(betas)
        z = fft.ifft(tmp, axis=2)

        return z

    def _step1(self, z):
        """
        Adjoint of _step1_t and final step of the forward transformation (coefficients
            to images).
        Performs the NUFFT on Fourier-space images to compute real-space images.
        """
        num_img = z.shape[0]
        z = z[:, :, : self.num_angular_nodes // 2].reshape(num_img, -1)
        im = anufft(
            z,
            np.stack((self.grid_x, self.grid_y)),
            (self.nres, self.nres),
            epsilon=self.epsilon,
        )
        im = im + np.conj(im)
        im = np.real(im)
        im = im.reshape(num_img, self.nres, self.nres)
        im[:, self.radial_mask] = 0

        return im

    def create_dense_matrix(self):
        """
        Directly computes the transformation matrix from Cartesian coordinates to
        FLE coordinates without any shortcuts.
        :return: A NumPy array of size `(self.nres**2, self.count)` containing the matrix
            entries.
        """
        # See Eqns. 3 and 4, Section 1.2
        ts = np.arctan2(self.ys, self.xs)

        B = np.zeros((self.nres, self.nres, self.count), dtype=np.complex128, order="F")
        for i in range(self.count):
            B[:, :, i] = self.basis_functions[i](self.rs, ts) * self.h
        B = B.reshape(self.nres**2, self.count)
        B = transform_complex_to_real(np.conj(B), self.ells)
        B = B.reshape(self.nres**2, self.count)
        if self.match_fb:
            flip_signs_indices = np.where(self.indices()["sgns"] == 1)
            B[:, flip_signs_indices] *= -1.0
            B = B[:, self.fb_compat_indices]
        return B

    def lowpass(self, coeffs, bandlimit):
        """
        Apply a low-pass filter to FLE coefficients `coeffs` with threshold `bandlimit`.
        :param coeffs: A NumPy array of FLE coefficients, of shape (num_images, self.count)
        :param bandlimit: Integer bandlimit (max frequency).
        :return: Band-limited coefficient array.
        """
        if len(coeffs.shape) == 1:
            coeffs = coeffs.reshape((1, coeffs.shape[0]))
        assert (
            len(coeffs.shape) == 2
        ), "Input a stack of coefficients of dimension (num_images, self.count)."
        assert (
            coeffs.shape[1] == self.count
        ), "Number of coefficients must match self.count."

        k = self.count - 1
        for _ in range(self.count):
            if self.bessel_zeros[k] / (np.pi) > (bandlimit - 1) // 2:
                k = k - 1
        coeffs[:, k + 1 :] = 0

        return coeffs

    def rotate(self, coeffs, theta):
        """
        Compute FLE coefficients for rotating an image by an angle `theta`.
        :param coeffs: A NumPy array of FLE coefficients of shape (num_images, self.count)
            to be rotated.
        :param theta: An angle in radians.
        :return: Rotated coefficient stack.
        """
        if len(coeffs.shape) == 1:
            coeffs = coeffs.reshape((1, self.count))
        assert (
            len(coeffs.shape) == 2
        ), "Input a stack of coefficients of dimension (num_images, self.count)."
        assert (
            coeffs.shape[1] == self.count
        ), "Number of coefficients must match self.count."

        coeffs_rot = np.zeros(coeffs.shape)
        num_img = coeffs.shape[0]
        for k in range(num_img):
            _coeffs = coeffs[k, :]
            b = np.zeros(self.count, dtype=np.complex128)
            for i in range(self.count):
                b[i] = np.exp(1j * theta * self.ells[i])
            b = b.flatten()
            coeffs_rot[k, :] = np.real(self.c2r @ (b * (self.r2c @ _coeffs).flatten()))

        return coeffs_rot

    def radialconv(self, coeffs, radial_img):
        """
        Convolve a stack of FLE coefficients with a 2D radial function.
        :param coeffs: A NumPy array of FLE coefficients of size (num_images, self.count).
        :param radial_img: A 2D NumPy array of size (self.nres, self.nres).
        :return: Convolved FLE coefficients.
        """
        num_img = coeffs.shape[0]
        coeffs_conv = np.zeros(coeffs.shape)
        for k in range(num_img):
            _coeffs = coeffs[k, :]
            z = self._step1_t(radial_img)
            b = self._step2_t(z)
            weights = self._radialconv_weights(b)
            b = weights / (self.h**2)
            b = b.reshape(self.count)
            coeffs_conv[k, :] = np.real(self.c2r @ (b * (self.r2c @ _coeffs).flatten()))

        return coeffs_conv

    def _radialconv_weights(self, b):
        """
        Helper function for step 3 of convolving with a radial function.
        """
        b = np.squeeze(b)
        b = np.array(b, order="F")
        if self.num_interp > self.num_radial_nodes:
            b = dct(b, axis=0, type=2) / (2 * self.num_radial_nodes)
            bz = np.zeros(b.shape)
            b = np.concatenate((b, bz), axis=0)
            b = idct(b, axis=0, type=2) * 2 * b.shape[0]
        a = np.zeros(self.count, dtype=np.float64)
        y = [None] * (self.ell_p_max + 1)
        for i in range(self.ell_p_max + 1):
            y[i] = (self.A3[i] @ b[:, 0]).flatten()
        for i in range(self.ell_p_max + 1):
            a[self.idx_list[i]] = y[i]

        return a.flatten()