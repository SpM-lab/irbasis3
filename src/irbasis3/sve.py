# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
from warnings import warn
import numpy as np

from . import _gauss
from . import poly
from . import svd
from . import kernel


def compute(K, eps=None, n_sv=None, n_gauss=None, dtype=float, work_dtype=None,
            sve_strat=None, svd_strat=None):
    """Perform truncated singular value expansion of a kernel.

    Perform a truncated singular value expansion (SVE) of a integral
    kernel `K : [xmin, xmax] x [ymin, ymax] -> R`:

        K(x, y) == sum(s[l] * u[l](x) * v[l](y) for l in (0, 1, 2, ...)),

    where `s[l]` are the singular values, which are ordered in non-increasing
    fashion, `u[l](x)` are the left singular functions, which form an
    orthonormal system on `[xmin, xmax]`, and `v[l](y)` are the right singular
    functions, which for an orthonormal system on `[ymin, ymax]`

    The SVE is mapped onto the singular value decomposition (SVD) of a matrix
    by expanding the kernel in piecewise Legendre polynomials (by default by
    using a collocation).

    Arguments:

      - `K`: Integral kernel to take SVE from
      - `eps`:  Relative cutoff for the singular values.  Defaults to double
        precision (2.2e-16) if the xprec package is available, and 1.5e-8
        otherwise.
      - `n_sv`: Maximum basis size.  If given, only at most the `n_sv` most
        significant singular values and associated singular functions are
        returned.
      - `n_gauss`: Order of Legendre polynomials.  Defaults to hinted value
        by the kernel.
      - `dtype`: Data type of the result.
      - `work_dtype`: Working data type.  Defaults to a data type with machine
        epsilon of at least `eps**2`, or otherwise most accurate data type
        available.
      - `sve_strat`: SVE to SVD translation strategy.  Defaults to SamplingSVE.
      - `svd_strat`: SVD solver. Defaults to fast (ID/RRQR) based solution
         when accuracy goals are moderate, and more accurate Jacobi-based
         algorithm otherwise.

    Return tuple `(u, s, v)`, where:

     - `u` is a `PiecewisePoly` instance holding the left singular functions
     - `s` is a vector of singular values
     - `v` is a `PiecewisePoly` instance holding the right singular functions
    """
    eps, work_dtype, default_svd_strat = _choose_accuracy(eps, work_dtype)
    if svd_strat is None:
        svd_strat = default_svd_strat
    if sve_strat is None:
        sve_strat = CentrosymmSVE if K.is_centrosymmetric else SamplingSVE
    sve = sve_strat(K, eps, n_gauss, work_dtype)
    matrix = sve.matrix
    u, s, v = svd.compute(matrix, sve.nsvals_hint, svd_strat)
    u, s, v = svd.truncate(u, s, v, eps, n_sv)
    return sve.postprocess(u, s, v, dtype)


class SamplingSVE:
    """SVE to SVD translation by sampling technique [1].

    Maps the singular value expansion (SVE) of a kernel `K` onto the singular
    value decomposition of a matrix `A`.  This is achieved by chosing two sets
    of Gauss quadrature rules: `(x, wx)` and `(y, wy)` and approximating the
    integrals in the SVE equations by finite sums.  This implies that the
    singular values of the SVE are well-approximated by the singular values of
    the following matrix:

        A[i, j] = sqrt(wx[i]) * K(x[i], y[j]) * sqrt(wy[j])

    and the values of the singular functions at the Gauss sampling points can
    be reconstructed from the singular vectors `u` and `v` as follows:

        u[l,i] ≈ sqrt(wx[i]) u[l](x[i])
        v[l,j] ≈ sqrt(wy[j]) u[l](y[j])

    [1] P. Hansen, Discrete Inverse Problems, Ch. 3.1
    """
    def __init__(self, K, eps, n_gauss=None, dtype=float):
        self.K = K
        sve_hints = K.hints(eps)
        if n_gauss is None:
            n_gauss = sve_hints.ngauss

        self.n_gauss = n_gauss
        self.nsvals_hint = sve_hints.nsvals
        self._rule = _gauss.legendre(n_gauss, dtype)
        self._segs_x = sve_hints.segments_x.astype(dtype)
        self._segs_y = sve_hints.segments_y.astype(dtype)
        self._gauss_x = self._rule.piecewise(self._segs_x)
        self._gauss_y = self._rule.piecewise(self._segs_y)
        self._sqrtw_x = np.sqrt(self._gauss_x.w)
        self._sqrtw_y = np.sqrt(self._gauss_y.w)

    @property
    def matrix(self):
        """SVD problem underlying the SVE."""
        result = kernel.matrix_from_gauss(self.K, self._gauss_x, self._gauss_y)
        result *= self._sqrtw_x[:, None]
        result *= self._sqrtw_y[None, :]
        return result

    def postprocess(self, u, s, v, dtype=None):
        """Constructs the SVE result from the SVD"""
        if dtype is None:
            dtype = np.result_type(u, s, v)

        s = s.astype(dtype)
        u_x = u / self._sqrtw_x[:,None]
        v_y = v / self._sqrtw_y[:,None]

        u_x = u_x.reshape(self._segs_x.size - 1, self.n_gauss, s.size)
        v_y = v_y.reshape(self._segs_y.size - 1, self.n_gauss, s.size)

        cmat = _gauss.legendre_collocation(self._rule)
        # lx,ixs -> ils -> lis
        u_data = (cmat @ u_x).transpose(1, 0, 2)
        v_data = (cmat @ v_y).transpose(1, 0, 2)

        dsegs_x = self._segs_x[1:] - self._segs_x[:-1]
        dsegs_y = self._segs_y[1:] - self._segs_y[:-1]
        u_data *= np.sqrt(.5 * dsegs_x)[None,:,None]
        v_data *= np.sqrt(.5 * dsegs_y)[None,:,None]

        # Construct polynomial
        ulx = poly.PiecewiseLegendrePoly(
                        u_data.astype(dtype), self._segs_x.astype(dtype))
        vly = poly.PiecewiseLegendrePoly(
                        v_data.astype(dtype), self._segs_y.astype(dtype))
        _canonicalize(ulx, vly)
        return ulx, s, vly


class CentrosymmSVE:
    """SVE of centrosymmetric kernel in block-diagonal (even/odd) basis.

    For a centrosymmetric kernel `K`, i.e., a kernel satisfying:
    `K(x, y) == K(-x, -y)`, one can make the following ansatz for the
    singular functions:

        u[l](x) = ured[l](x) + sign[l] * ured[l](-x)
        v[l](y) = vred[l](y) + sign[l] * ured[l](-y)

    where `sign[l]` is either +1 or -1.  This means that the singular value
    expansion can be block-diagonalized into an even and an odd part by
    (anti-)symmetrizing the kernel:

        Keven = K(x, y) + K(x, -y)
        Kodd  = K(x, y) - K(x, -y)

    The l-th basis function, restricted to the positive interval, is then
    the singular function of one of these kernels.  If the kernel generates a
    Chebyshev system [1], then even and odd basis functions alternate.

    [1]: A. Karlin, Total Positivity (1968).
    """
    def __init__(self, K, eps, n_gauss=None, dtype=float, InnerSVE=None):
        if InnerSVE is None:
            InnerSVE = SamplingSVE
        self.K = K

        # Inner kernels for even and odd functions
        self.even = InnerSVE(K.get_symmetrized(+1), eps, n_gauss, dtype)
        self.odd = InnerSVE(K.get_symmetrized(-1), eps, n_gauss, dtype)

        # Now extract the hints
        self.nsvals_hint = max(self.even.nsvals_hint, self.odd.nsvals_hint)

    @property
    def matrix(self):
        """Set of SVD problems underlying the SVE."""
        return np.array([self.even.matrix, self.odd.matrix])

    def postprocess(self, u, s, v, dtype):
        """Constructs the SVE result from the SVD"""
        u_even, s_even, v_even = self.even.postprocess(u[0], s[0], v[0], dtype)
        u_odd, s_odd, v_odd = self.odd.postprocess(u[1], s[1], v[1], dtype)

        # Merge two sets - data is [legendre, segment, l]
        u_data = np.concatenate([u_even.data, u_odd.data], axis=2)
        v_data = np.concatenate([v_even.data, v_odd.data], axis=2)
        s = np.concatenate([s_even, s_odd])
        signs = np.concatenate([np.ones(s_even.size), -np.ones(s_odd.size)])

        # Sort: now for totally positive kernels like defined in this module,
        # this strictly speaking is not necessary as we know that the even/odd
        # functions intersperse.
        sort = s.argsort()[::-1]
        u_data = u_data[:, :, sort]
        v_data = v_data[:, :, sort]
        s = s[sort]
        signs = signs[sort]

        # Extend to the negative side
        inv_sqrt2 = 1/np.sqrt(2, dtype=u_data.dtype)
        u_data *= inv_sqrt2
        v_data *= inv_sqrt2
        poly_flip_x = ((-1)**np.arange(u_data.shape[0]))[:, None, None]
        u_neg = u_data[:, ::-1, :] * poly_flip_x * signs
        v_neg = v_data[:, ::-1, :] * poly_flip_x * signs
        u_data = np.concatenate([u_neg, u_data], axis=1)
        v_data = np.concatenate([v_neg, v_data], axis=1)

        # TODO: this relies on specific symmetrization behaviour ...
        full_hints = self.K.hints(None)
        u = poly.PiecewiseLegendrePoly(u_data, full_hints.segments_x)
        v = poly.PiecewiseLegendrePoly(v_data, full_hints.segments_y)
        return u, s, v


def _choose_accuracy(eps, work_dtype):
    """Choose work dtype and accuracy based on specs and defaults"""
    if eps is None:
        if work_dtype is None:
            return np.sqrt(svd.MAX_EPS), svd.MAX_DTYPE, 'fast'
        safe_eps = np.sqrt(svd.finfo(work_dtype).eps)
        return safe_eps, work_dtype, 'fast'

    if work_dtype is None:
        if eps >= np.sqrt(svd.finfo(float).eps):
            return eps, float, 'fast'
        work_dtype = svd.MAX_DTYPE

    safe_eps = np.sqrt(svd.finfo(work_dtype).eps)
    if eps >= safe_eps:
        svd_strat = 'fast'
    else:
        svd_strat = 'accurate'
        msg = ("\nBasis cutoff is {:.2g}, which is below sqrt(eps) with\n"
               "eps = {:.2g}.  Expect singular values and basis functions\n"
               "for large l to have lower precision than the cutoff.\n")
        msg = msg.format(float(eps), float(np.square(safe_eps)))
        if svd._ddouble is None:
            msg += "You can install the xprec package to gain more precision.\n"
        warn(msg, UserWarning, 3)

    return eps, work_dtype, svd_strat


def _canonicalize(ulx, vly):
    """Canonicalize basis.

    Each SVD (u_l, v_l) pair is unique only up to a global phase, which may
    differ from implementation to implementation and also platform.  We
    fix that gauge by demanding u_l(1) > 0.  This ensures a diffeomorphic
    connection to the Legendre polynomials for lambda_ -> 0.
    """
    gauge = np.sign(ulx(1))
    ulx.data[None, None, :] *= 1/gauge
    vly.data[None, None, :] *= gauge