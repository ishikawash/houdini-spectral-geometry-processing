import sys
import numpy as np
import numpy.typing as npt
from scipy.sparse import lil_matrix, dia_matrix
from MH._types import SpectralFilterFunction

try:
    import cupy as cp
    from cupyx.scipy.sparse import linalg as cpx_linalg
except:
    pass


def cupy_available() -> bool:
    return "cupy" in sys.modules

def make_spectral_transform_matrix(
    V: npt.NDArray[np.float64],
    M: npt.NDArray[np.float64],
    tolerance: float
) -> lil_matrix:
    M_ = cp.array(M)
    V_ = cp.array(V)
    A_ = M_ @ V_.T # (N,k)
    mask = cp.abs(A_) <= tolerance
    A_[mask] = 0.0
    return lil_matrix(A_.get())

def make_spectral_filter_matrix(
    W: npt.NDArray[np.float64],
    V: npt.NDArray[np.float64],
    M: npt.NDArray[np.float64],
    tolerance: float,
    filter: SpectralFilterFunction
) -> lil_matrix:
    M_ = cp.array(M)
    V_ = cp.array(V)
    h_ = cp.array(filter(W)) # (1,k)
    A_ = M_ @ V_.T # (N,k)
    f_ = h_ * A_
    A_ = f_ @ V_ # (N,N)
    mask = cp.abs(A_) <= tolerance
    A_[mask] = 0.0
    return lil_matrix(A_.get())


def solve_eigenvalue_problem(
    L: lil_matrix,
    M: dia_matrix,
    k: int = 1,
    tolerance: float = 1e-6
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    L_ = cp.array(L.toarray())
    M_ = cp.linalg.inv(cp.sqrt(cp.array(M.toarray())))
    LM_ = M_ @ L_ @ M_
    W_, V_ = cpx_linalg.eigsh(LM_, k=k, which='SA', tol=tolerance)
    W = W_.get()
    V = ((M_ @ V_).T).get()
    return W, V
