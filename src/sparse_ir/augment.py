# Copyright (C) 2020-2021 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np

from . import _util
from . import abstract
from . import basis


class AugmentedBasis(abstract.AbstractBasis):
    """Augmented basis on the imaginary-time/frequency axis.

    Groups a set of additional functions, ``augmentations``, with a given
    ``basis``.  The augmented functions then form the first basis
    functions, while the rest is provided by the regular basis, i.e.::

        u[l](x) == augmentations[l](x) if l < naug else basis.u[l-naug](x),

    where ``naug = len(augmentations)`` is the number of added basis functions
    through augmentation.  Similar expressions hold for Matsubara frequencies.

    Augmentation is useful in constructing bases for vertex-like quantities
    such as self-energies `[1]`_.  It is also useful when constructing a
    two-point kernel that serves as a base for multi-point functions `[2]`_.

    Example:
        For constructing the vertex basis and the augmented basis, one can
        use::

            import sparse_ir, sparse_ir.augment as aug
            basis = sparse_ir.FiniteTempBasis('B', beta=10, wmax=2.0)
            vertex_basis = aug.AugmentedBasis(basis, aug.MatsubaraConst)
            aug_basis = aug.AugmentedBasis(basis, aug.TauConst, aug.TauLinear)

    Warning:
        Bases augmented with `TauConst` and `TauLinear` tend to be poorly
        conditioned.  Care must be taken while fitting and compactness should
        be enforced if possible to regularize the problem.

        While vertex bases, i.e., bases augmented with `MatsubaraConst`, stay
        reasonably well-conditioned, it is still good practice to treat the
        Hartree--Fock term separately rather than including it in the basis,
        if possible.

    See also:
         - :class:`MatsubaraConst` for vertex basis `[1]`_
         - :class:`TauConst`, :class:`TauLinear` for multi-point `[2]`_

    .. _[1]: https://doi.org/10.1103/PhysRevResearch.3.033168
    .. _[2]: https://doi.org/10.1103/PhysRevB.97.205111
    """
    def __init__(self, basis, *augmentations):
        augmentations = tuple(_augmentation_factory(basis, *augmentations))
        self._basis = basis
        self._augmentations = augmentations
        self._naug = len(augmentations)

        self._u = AugmentedTauFunction(self._basis.u, augmentations)
        self._uhat = AugmentedMatsubaraFunction(
                        self._basis.uhat, [aug.hat for aug in augmentations])

    @property
    def u(self):
        return self._u

    @property
    def uhat(self):
        return self._uhat

    @property
    def statistics(self):
        raise self._basis.statistics

    def __getitem__(self, index):
        stop = basis._slice_to_size(index)
        if stop <= self._naug:
            raise ValueError("Cannot truncate to only augmentation")
        return AugmentedBasis(self._basis[:stop - self._naug],
                              *self._augmentations)

    @property
    def shape(self):
        return self.size,

    @property
    def size(self):
        return self._naug + self._basis.size

    @property
    def significance(self):
        return self._basis.significance

    @property
    def accuracy(self):
        return self._basis.accuracy

    @property
    def lambda_(self):
        return self._basis.lambda_

    @property
    def beta(self):
        return self._basis.beta

    @property
    def wmax(self):
        return self._basis.wmax

    def default_tau_sampling_points(self, *, npoints=None):
        if npoints is None:
            npoints = self.size

        # Return the sampling points of the underlying basis, but since we
        # use the size of self, we add two further points.  One then has to
        # hope that these give good sampling points.
        return self._basis.default_tau_sampling_points(npoints=npoints)

    def default_matsubara_sampling_points(self, *, npoints=None,
                                          positive_only=False):
        if npoints is None:
            npoints = self.size
        return self._basis.default_matsubara_sampling_points(
                                npoints=npoints, positive_only=positive_only)

    @property
    def is_well_conditioned(self):
        wbasis = self._basis.is_well_conditioned
        waug = (len(self._augmentations) == 1
                and isinstance(self._augmentations[0], MatsubaraConst))
        return wbasis and waug


class _AugmentedFunction:
    def __init__(self, fbasis, faug):
        if fbasis.ndim != 1:
            raise ValueError("must have vector of functions as fbasis")
        self._fbasis = fbasis
        self._faug = faug
        self._naug = len(faug)

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        return self.size,

    @property
    def size(self):
        return self._naug + self._fbasis.size

    def __call__(self, x):
        x = np.asarray(x)
        fbasis_x = self._fbasis(x)
        faug_x = [faug_l(x)[None] for faug_l in self._faug]
        f_x = np.concatenate(faug_x + [fbasis_x], axis=0)
        assert f_x.shape[1:] == x.shape
        return f_x

    def __getitem__(self, l):
        # TODO make this more general
        if isinstance(l, slice):
            stop = basis._slice_to_size(l)
            if stop <= self._naug:
                raise NotImplementedError("Don't truncate to only augmentation")
            return _AugmentedFunction(self._fbasis[:stop-self._naug], self._faug)
        else:
            l = int(l)
            if l < self._naug:
                return self._faug[l]
            else:
                return self._fbasis[l-self._naug]


class AugmentedTauFunction(_AugmentedFunction):
    @property
    def xmin(self):
        return self._fbasis.xmin

    @property
    def xmax(self):
        return self._fbasis.xmin

    def deriv(self, n=1):
        """Get polynomial for the n'th derivative"""
        dbasis = self._fbasis.deriv(n)
        daug = [faug_l.deriv(n) for faug_l in self._faug]
        return AugmentedTauFunction(dbasis, *daug)


class AugmentedMatsubaraFunction(_AugmentedFunction):
    @property
    def zeta(self):
        return self._fbasis.zeta


class AbstractAugmentation:
    """Scalar function in imaginary time/frequency.

    This represents a single function in imaginary time and frequency,
    together with some auxiliary methods that make it suitable for augmenting
    a basis.

    See also:
        :class:`AugmentedBasis`
    """
    @classmethod
    def create(cls, basis):
        """Factory method constructing an augmented term for a basis"""
        raise NotImplementedError()

    def __call__(self, tau):
        """Evaluate the function at imaginary time ``tau``"""
        raise NotImplementedError()

    def deriv(self, n):
        """Derivative of order ``n`` of the function"""
        raise NotImplementedError()

    def hat(self, n):
        """Evaluate the Fourier transform at reduced frequency ``n``"""
        raise NotImplementedError()


class TauConst(AbstractAugmentation):
    """Constant in imaginary time/discrete delta in frequency"""
    @classmethod
    def create(cls, basis):
        _check_bosonic_statistics(basis.statistics)
        return cls(basis.beta)

    def __init__(self, beta):
        if beta <= 0:
            raise ValueError("temperature must be positive")
        self._beta = beta

    def __call__(self, tau):
        tau = _util.check_range(tau, 0, self._beta)
        return np.broadcast_to(1 / np.sqrt(self._beta), tau.shape)

    def deriv(self, n=1):
        if n == 0:
            return self
        else:
            return lambda tau: np.zeros_like(tau)

    def hat(self, n):
        n = _util.check_reduced_matsubara(n, zeta=0)
        return np.sqrt(self._beta) * (n == 0).astype(complex)


class TauLinear(AbstractAugmentation):
    """Linear function in imaginary time, antisymmetric around beta/2"""
    @classmethod
    def create(cls, basis):
        _check_bosonic_statistics(basis.statistics)
        return cls(basis.beta)

    def __init__(self, beta):
        if beta <= 0:
            raise ValueError("temperature must be positive")
        self._beta = beta
        self._norm = np.sqrt(3/beta)

    def __call__(self, tau):
        tau = _util.check_range(tau, 0, self._beta)
        x = 2/self._beta * tau - 1
        return self._norm * x

    def deriv(self, n=1):
        if n == 0:
            return self
        elif n == 1:
            c = self._norm * 2/self._beta
            return lambda tau: np.full_like(tau, c)
        else:
            return lambda tau: np.zeros_like(tau)

    def hat(self, n):
        n = _util.check_reduced_matsubara(n, zeta=0)
        inv_w = np.pi/self._beta * n
        inv_w = np.reciprocal(inv_w, out=inv_w, where=n.astype(bool))
        return self._norm * 2/1j * inv_w


class MatsubaraConst(AbstractAugmentation):
    """Constant in Matsubara, undefined in imaginary time"""
    @classmethod
    def create(cls, basis):
        return cls(basis.beta)

    def __init__(self, beta):
        if beta <= 0:
            raise ValueError("temperature must be positive")
        self._beta = beta

    def __call__(self, tau):
        tau = _util.check_range(tau, 0, self._beta)
        return np.broadcast_to(np.nan, tau.shape)

    def deriv(self, n=1):
        return self

    def hat(self, n):
        n = _util.check_reduced_matsubara(n)
        return np.broadcast_to(1.0, n.shape)


def _augmentation_factory(basis, *augs):
    for aug in augs:
        if isinstance(aug, AbstractAugmentation):
            yield aug
        else:
            yield aug.create(basis)


def _check_bosonic_statistics(statistics):
    if statistics == 'B':
        return
    elif statistics == 'F':
        raise ValueError("term only allowed for bosonic basis")
    else:
        raise ValueError("invalid statistics")
