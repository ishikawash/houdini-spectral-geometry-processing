import numpy as np
import numpy.typing as npt
from scipy.sparse import lil_matrix, dia_matrix
from scipy.sparse import linalg as spy_linalg
from MH._types import SpectralFilterFunction


def make_spectral_transform_matrix(
    V: npt.NDArray[np.float64],
    M: dia_matrix,
    tolerance: float
) -> lil_matrix:
    A = M @ V.T # (N,k)
    mask = np.abs(A) <= tolerance
    A[mask] = 0.0
    return lil_matrix(A)

def make_spectral_filter_matrix(
    W: npt.NDArray[np.float64],
    V: npt.NDArray[np.float64],
    M: dia_matrix,
    tolerance: float,
    filter: SpectralFilterFunction
) -> lil_matrix:
    A = M @ V.T # (N,k)
    h = filter(W) # (1,k)
    f = h * A
    A_ = f @ V # (N,N)
    mask = np.abs(A_) <= tolerance
    A_[mask] = 0.0
    return lil_matrix(A_)

def solve_eigenvalue_problem(
    L: lil_matrix,
    M: dia_matrix,
    k: int = 1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    W, V = spy_linalg(L.tocsr(), k=k, M=M, sigma=0)
    return W, V.T
