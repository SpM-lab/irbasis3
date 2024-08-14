
"""
Intermediate representation (IR) for many-body propagators
==========================================================

This library provides routines for constructing and working with the
intermediate representation of correlation functions.  It provides:

 - on-the-fly computation of basis functions for arbitrary cutoff Λ
 - basis functions and singular values are accurate to full precision
 - routines for sparse sampling
"""
__copyright__ = "2020-2024 Markus Wallerberger, Hiroshi Shinaoka, and others"
__license__ = "MIT"
__version__ = "1.1.2"

from .kernel import RegularizedBoseKernel, LogisticKernel
from .sve import compute as compute_sve, SVEResult
from .basis import FiniteTempBasis, finite_temp_bases
from .basis_set import FiniteTempBasisSet
from .sampling import TauSampling, MatsubaraSampling
