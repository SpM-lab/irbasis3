# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np
from warnings import warn
import numpy.polynomial.legendre as np_legendre
import scipy.special as sp_special

from . import _util
from . import _roots
from . import _gauss


class PiecewiseLegendrePoly:
    """Piecewise Legendre polynomial.

    Models a function on the interval ``[-1, 1]`` as a set of segments on the
    intervals ``S[i] = [a[i], a[i+1]]``, where on each interval the function
    is expanded in scaled Legendre polynomials.
    """
    def __init__(self, data, knots, dx=None, symm=None):
        """Piecewise Legendre polynomial"""
        if np.isnan(data).any():
            raise ValueError("PiecewiseLegendrePoly: data contains NaN!")
        if isinstance(knots, self.__class__):
            if dx is not None or symm is None:
                raise RuntimeError("wrong arguments")
            self.__dict__.update(knots.__dict__)
            self.data = data
            self.symm = symm
            return

        data = np.array(data)
        knots = np.array(knots)
        polyorder, nsegments = data.shape[:2]
        if knots.shape != (nsegments+1,):
            raise ValueError("Invalid knots array")
        if not (knots[1:] >= knots[:-1]).all():
            raise ValueError("Knots must be monotonically increasing")
        if symm is None:
            # TODO: infer symmetry from data
            symm = np.zeros(data.shape[2:])
        else:
            symm = np.array(symm)
            if symm.shape != data.shape[2:]:
                raise ValueError("shape mismatch")
        if dx is None:
            dx = knots[1:] - knots[:-1]
        else:
            dx = np.array(dx)
            if not np.allclose(dx, knots[1:] - knots[:-1]):
                raise ValueError("dx must work with knots")

        self.nsegments = nsegments
        self.polyorder = polyorder
        self.xmin = knots[0]
        self.xmax = knots[-1]

        self.knots = knots
        self.dx = dx
        self.data = data
        self.symm = symm
        self._xm = .5 * (knots[1:] + knots[:-1])
        self._inv_xs = 2/dx
        self._norm = np.sqrt(self._inv_xs)

    def __getitem__(self, l):
        """Return part of a set of piecewise polynomials"""
        new_symm = self.symm[l]
        if isinstance(l, tuple):
            new_data = self.data[(slice(None), slice(None), *l)]
        else:
            new_data = self.data[:,:,l]
        return self.__class__(new_data, self, symm=new_symm)

    def __call__(self, x):
        """Evaluate polynomial at position x"""
        i, xtilde = self._split(np.asarray(x))
        data = self.data[:, i]

        # Evaluate for all values of l.  x and data array must be
        # broadcast'able against each other, so we append dimensions here
        func_dims = self.data.ndim - 2
        datashape = i.shape + (1,) * func_dims
        res = np_legendre.legval(xtilde.reshape(datashape), data, tensor=False)
        res *= self._norm[i.reshape(datashape)]

        # Finally, exchange the x and vector dimensions
        order = tuple(range(i.ndim, i.ndim + func_dims)) + tuple(range(i.ndim))
        return res.transpose(*order)

    def value(self, l, x):
        """Return value for l and x."""
        if self.data.ndim != 3:
            raise ValueError("Only allowed for vector of data")

        l, x = np.broadcast_arrays(l, x)
        i, xtilde = self._split(x)
        data = self.data[:, i, l]

        # This should now neatly broadcast against each other
        res = np_legendre.legval(xtilde, data, tensor=False)
        res *= self._norm[i]
        return res

    def overlap(self, f, *, rtol=2.3e-16, return_error=False, points=None):
        r"""Evaluate overlap integral of this polynomial with function ``f``.

        Given the function ``f``, evaluate the integral::

            ∫ dx * f(x) * self(x)

        using piecewise Gauss-Legendre quadrature, where ``self`` are the
        polynomials.

        Arguments:
            f (callable):
                function that is called with a point ``x`` and returns ``f(x)``
                at that position.

            points (sequence of floats)
                A sequence of break points in the integration interval
                where local difficulties of the integrand may occur
                (e.g., singularities, discontinuities)

        Return:
            array-like object with shape (poly_dims, f_dims)
            poly_dims are the shape of the polynomial and f_dims are those
            of the function f(x).
        """
        int_result, int_error = _compute_overlap(self, f, rtol=rtol, points=points)
        if return_error:
            return int_result, int_error
        else:
            return int_result

    def deriv(self, n=1):
        """Get polynomial for the n'th derivative"""
        ddata = np_legendre.legder(self.data, n)

        _scale_shape = (1, -1) + (1,) * (self.data.ndim - 2)
        scale = self._inv_xs ** n
        ddata *= scale.reshape(_scale_shape)
        return self.__class__(ddata, self, symm=(-1)**n * self.symm)

    def roots(self, alpha=2):
        """Find all roots of the piecewise polynomial

        Assume that between each two knots (pieces) there are at most ``alpha``
        roots.
        """
        if self.data.ndim > 2:
            raise ValueError("select single polynomial before calling roots()")

        grid = self.knots
        xmid = (self.xmax + self.xmin) / 2
        if self.symm:
            if grid[self.nsegments // 2] == xmid:
                grid = grid[self.nsegments//2:]
            else:
                grid = np.hstack((xmid, grid[grid > xmid]))

        grid = _refine_grid(grid, alpha)
        roots = _roots.find_all(self, grid)

        if self.symm == 1:
            revroots = (self.xmax + self.xmin) - roots[::-1]
            roots = np.hstack((revroots, roots))
        elif self.symm == -1:
            # There must be a zero at exactly the midpoint, but we may either
            # slightly miss it or have a spurious zero
            if roots.size:
                if roots[0] == xmid or self(xmid) * self.deriv()(xmid) < 0:
                    roots = roots[1:]
            revroots = (self.xmax + self.xmin) - roots[::-1]
            roots = np.hstack((revroots, xmid, roots))

        return roots

    @property
    def shape(self): return self.data.shape[2:]

    @property
    def size(self): return self.data[:1,:1].size

    @property
    def ndim(self): return self.data.ndim - 2

    def _split(self, x):
        """Split segment"""
        x = _util.check_range(x, self.xmin, self.xmax)
        i = self.knots.searchsorted(x, 'right').clip(None, self.nsegments)
        i -= 1
        xtilde = x - self._xm[i]
        xtilde *= self._inv_xs[i]
        return i, xtilde


class PiecewiseLegendreFT:
    """Fourier transform of a piecewise Legendre polynomial.

    For a given frequency index ``n``, the Fourier transform of the Legendre
    function is defined as::

            phat(n) == ∫ dx exp(1j * pi * n * x / (xmax - xmin)) p(x)

    The polynomial is continued either periodically (``freq='even'``), in which
    case ``n`` must be even, or antiperiodically (``freq='odd'``), in which case
    ``n`` must be odd.
    """
    _DEFAULT_GRID = np.hstack([np.arange(2**6),
                        (2**np.linspace(6, 35, 16*(35-6)+1)).astype(int)])

    def __init__(self, poly, freq='even', n_asymp=None, power_model=None):
        if poly.xmin != -1 or poly.xmax != 1:
            raise NotImplementedError("Only interval [-1, 1] supported")
        self.poly = poly
        self.freq = freq
        self.zeta = {'any': None, 'even': 0, 'odd': 1}[freq]
        if n_asymp is None:
            self.n_asymp = np.inf
            self._model = None
        else:
            self.n_asymp = n_asymp
            if power_model is None:
                self._model = _power_model(freq, poly)
            else:
                self._model = power_model

    @property
    def shape(self): return self.poly.shape

    @property
    def size(self): return self.poly.size

    @property
    def ndim(self): return self.poly.ndim

    def __getitem__(self, l):
        model = self._model if self._model is None else self._model[l]
        return self.__class__(self.poly[l], self.freq, self.n_asymp, model)

    @_util.ravel_argument(last_dim=True)
    def __call__(self, n):
        """Obtain Fourier transform of polynomial for given frequencies"""
        n = _util.check_reduced_matsubara(n, self.zeta)
        result = _compute_unl_inner(self.poly, n)

        # We use the asymptotics at frequencies larger than conv_radius
        # since it has lower relative error.
        cond_outer = np.abs(n) >= self.n_asymp
        if cond_outer.any():
            n_outer = n[cond_outer]
            result[..., cond_outer] = self._model.giw(n_outer).T

        return result

    def extrema(self, *, part=None, grid=None, positive_only=False):
        """Obtain extrema of Fourier-transformed polynomial."""
        if self.poly.shape:
            raise ValueError("select single polynomial")
        if grid is None:
            grid = self._DEFAULT_GRID

        f = self._func_for_part(part)
        x0 = _roots.discrete_extrema(f, grid)
        x0 = 2 * x0 + self.zeta
        if not positive_only:
            x0 = _symmetrize_matsubara(x0)
        return x0

    def sign_changes(self, *, part=None, grid=None, positive_only=False):
        """Obtain sign changes of Fourier-transformed polynomial."""
        if self.poly.shape:
            raise ValueError("select single polynomial")
        if grid is None:
            grid = self._DEFAULT_GRID

        f = self._func_for_part(part)
        x0 = _roots.find_all(f, grid, type='discrete')
        x0 = 2 * x0 + self.zeta
        if not positive_only:
            x0 = _symmetrize_matsubara(x0)
        return x0

    def _func_for_part(self, part=None):
        if part is None:
            parity = self.poly.symm
            if np.allclose(parity, 1):
                part = 'real' if self.zeta == 0 else 'imag'
            elif np.allclose(parity, -1):
                part = 'imag' if self.zeta == 0 else 'real'
            else:
                raise ValueError("cannot detect parity.")
        if part == 'real':
            return lambda n: self(2*n + self.zeta).real
        elif part == 'imag':
            return lambda n: self(2*n + self.zeta).imag
        else:
            raise ValueError("part must be either 'real' or 'imag'")



def _imag_power(n):
    """Imaginary unit raised to an integer power without numerical error"""
    n = np.asarray(n)
    if not np.issubdtype(n.dtype, np.integer):
        raise ValueError("expecting set of integers here")
    cycle = np.array([1, 0+1j, -1, 0-1j], complex)
    return cycle[n % 4]


def _get_tnl(l, w):
    r"""Fourier integral of the l-th Legendre polynomial::

        T_l(w) == \int_{-1}^1 dx \exp(iwx) P_l(x)
    """
    # spherical_jn gives NaN for w < 0, but since we know that P_l(x) is real,
    # we simply conjugate the result for w > 0 in these cases.
    result = 2 * _imag_power(l) * sp_special.spherical_jn(l, np.abs(w))
    np.conjugate(result, out=result, where=w < 0)
    return result


def _shift_xmid(knots, dx):
    r"""Return midpoint relative to the nearest integer plus a shift.

    Return the midpoints ``xmid`` of the segments, as pair ``(diff, shift)``,
    where shift is in ``(0,1,-1)`` and ``diff`` is a float such that
    ``xmid == shift + diff`` to floating point accuracy.
    """
    dx_half = dx / 2
    xmid_m1 = dx.cumsum() - dx_half
    xmid_p1 = -dx[::-1].cumsum()[::-1] + dx_half
    xmid_0 = knots[1:] - dx_half

    shift = np.round(xmid_0).astype(int)
    diff = np.choose(shift+1, (xmid_m1, xmid_0, xmid_p1))
    return diff, shift


def _phase_stable(poly, wn):
    """Phase factor for the piecewise Legendre to Matsubara transform.

    Compute the following phase factor in a stable way::

        np.exp(1j * np.pi/2 * wn[:,None] * poly.dx.cumsum()[None,:])
    """
    # A naive implementation is losing precision close to x=1 and/or x=-1:
    # there, the multiplication with `wn` results in `wn//4` almost extra turns
    # around the unit circle.  The cosine and sines will first map those
    # back to the interval [-pi, pi) before doing the computation, which loses
    # digits in dx.  To avoid this, we extract the nearest integer dx.cumsum()
    # and rewrite above expression like below.
    #
    # Now `wn` still results in extra revolutions, but the mapping back does
    # not cut digits that were not there in the first place.
    xmid_diff, extra_shift = _shift_xmid(poly.knots, poly.dx)

    if np.issubdtype(wn.dtype, np.integer):
        shift_arg = wn[None,:] * xmid_diff[:,None]
    else:
        delta_wn, wn = np.modf(wn)
        wn = wn.astype(int)
        shift_arg = wn[None,:] * xmid_diff[:,None]
        shift_arg += delta_wn[None,:] * (extra_shift + xmid_diff)[:,None]

    phase_shifted = np.exp(0.5j * np.pi * shift_arg)
    corr = _imag_power((extra_shift[:,None] + 1) * wn[None,:])
    return corr * phase_shifted


def _compute_unl_inner(poly, wn):
    """Compute piecewise Legendre to Matsubara transform."""
    dx_half = poly.dx / 2

    data_flat = poly.data.reshape(*poly.data.shape[:2], -1)
    data_sc = data_flat * np.sqrt(dx_half/2)[None,:,None]
    p = np.arange(poly.polyorder)

    wred = np.pi/2 * wn
    phase_wi = _phase_stable(poly, wn)
    t_pin = _get_tnl(p[:,None,None], wred[None,:] * dx_half[:,None]) * phase_wi

    # Perform the following, but faster:
    #   resulth = einsum('pin,pil->nl', t_pin, data_sc)
    npi = poly.polyorder * poly.nsegments
    result_flat = (t_pin.reshape(npi,-1).T @ data_sc.reshape(npi,-1)).T
    return result_flat.reshape(*poly.data.shape[2:], wn.size)


class _PowerModel:
    """Model from a high-frequency series expansion::

        A(iw) == sum(A[n] / (iw)**(n+1) for n in range(1, N))

    where ``iw == 1j * pi/2 * wn`` is a reduced imaginary frequency, i.e.,
    ``wn`` is an odd/even number for fermionic/bosonic frequencies.
    """
    def __init__(self, moments):
        """Initialize model"""
        if moments.ndim == 1:
            moments = moments[:, None]
        self.moments = np.asarray(moments)
        self.nmom, self.nl = self.moments.shape

    @_util.ravel_argument()
    def giw(self, wn):
        """Return model Green's function for vector of frequencies"""
        wn = _util.check_reduced_matsubara(wn)
        result_dtype = np.result_type(1j, wn, self.moments)
        result = np.zeros((wn.size, self.nl), result_dtype)
        inv_iw = 1j * np.pi/2 * wn
        np.reciprocal(inv_iw, out=inv_iw, where=(wn != 0))
        for mom in self.moments[::-1]:
            result += mom
            result *= inv_iw[:, None]
        return result

    def __getitem__(self, l):
        return self.__class__(self.moments[:,l])


def _derivs(ppoly, x):
    """Evaluate polynomial and its derivatives at specific x"""
    yield ppoly(x)
    for _ in range(ppoly.polyorder-1):
        ppoly = ppoly.deriv()
        yield ppoly(x)


def _power_moments(stat, deriv_x1):
    """Return moments"""
    statsign = {'odd': -1, 'even': 1}[stat]
    mmax, lmax = deriv_x1.shape
    m = np.arange(mmax)[:,None]
    l = np.arange(lmax)[None,:]
    coeff_lm = ((-1.0)**(m+1) + statsign * (-1.0)**l) * deriv_x1
    return -statsign/np.sqrt(2.0) * coeff_lm


def _power_model(stat, poly):
    deriv_x1 = np.asarray(list(_derivs(poly, x=1)))
    if deriv_x1.ndim == 1:
        deriv_x1 = deriv_x1[:,None]
    moments = _power_moments(stat, deriv_x1)
    return _PowerModel(moments)


def _refine_grid(knots, alpha):
    """Linear refinement of grid"""
    result = np.linspace(knots[:-1], knots[1:], alpha, endpoint=False)
    return np.hstack((result.T.ravel(), knots[-1]))


def _symmetrize_matsubara(x0):
    if not x0.size:
        return x0
    if not (x0[1:] >= x0[:-1]).all():
        raise ValueError("set of Matsubara points not ordered")
    if not (x0[0] >= 0):
        raise ValueError("points must be non-negative")
    if x0[0] == 0:
        x0 = np.hstack([-x0[::-1], x0[1:]])
    else:
        x0 = np.hstack([-x0[::-1], x0])
    return x0


def _compute_overlap(poly, f, rtol=2.3e-16, radix=2, max_refine_levels=40,
                     max_refine_points=2000, points=None):
    base_rule = _gauss.kronrod_31_15()
    if points is None:
        knots = poly.knots
    else:
        points = np.asarray(points)
        knots = np.unique(np.hstack((poly.knots, points)))
    xstart = knots[:-1]
    xstop = knots[1:]

    f_shape = None
    res_value = 0
    res_error = 0
    res_magn = 0
    for _ in range(max_refine_levels):
        #print(f"Level {_}: {xstart.size} segments")
        if xstart.size > max_refine_points:
            warn("Refinement is too broad, aborting (increase rtol)")
            break

        rule = base_rule.reseat(xstart[:, None], xstop[:, None])

        fx = np.array(list(map(f, rule.x.ravel())))
        if f_shape is None:
            f_shape = fx.shape[1:]
        elif fx.shape[1:] != f_shape:
            raise ValueError("inconsistent shapes")
        fx = fx.reshape(rule.x.shape + (-1,))

        valx = poly(rule.x).reshape(-1, *rule.x.shape, 1) * fx
        int21 = (valx[:, :, :, :] * rule.w[:, :, None]).sum(2)
        int10 = (valx[:, :, rule.vsel, :] * rule.v[:, :, None]).sum(2)
        intdiff = np.abs(int21 - int10)
        intmagn = np.abs(int10)

        magn = res_magn + intmagn.sum(1).max(1)
        relerror = intdiff.max(2) / magn[:, None]

        xconverged = (relerror <= rtol).all(0)
        res_value += int10[:, xconverged].sum(1)
        res_error += intdiff[:, xconverged].sum(1)
        res_magn += intmagn[:, xconverged].sum(1).max(1)
        if xconverged.all():
            break

        xrefine = ~xconverged
        xstart = xstart[xrefine]
        xstop = xstop[xrefine]
        xedge = np.linspace(xstart, xstop, radix + 1, axis=-1)
        xstart = xedge[:, :-1].ravel()
        xstop = xedge[:, 1:].ravel()
    else:
        warn("Integration did not converge after refinement")

    res_shape = poly.shape + f_shape
    return res_value.reshape(res_shape), res_error.reshape(res_shape)
