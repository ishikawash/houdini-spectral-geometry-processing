from typing import Any, Generator, Callable, Optional
import os
import dataclasses
import cProfile
from pathlib import Path
import numpy as np
import numpy.typing as npt
from scipy.sparse import lil_matrix, dia_matrix, diags
from scipy.sparse.linalg import eigsh
import hou


@dataclasses.dataclass(frozen=True)
class LaplacianRow:
    mass: float
    column_indices: list[int]
    column_values: list[float]

class NodeParameters:
    def __init__(self, node: hou.Node):
        self._node = node

    @property
    def eigenvalue_num(self) -> int:
        N: int = self._node.geometry().pointCount()
        x: int = self._node.parm("eigenvalue_num").eval()
        return max(min(x, N - 1), 1)

    @property
    def tolerance(self) -> float:
        return self._node.parm("tolerance").eval()

HouPointIterator = Generator[hou.Point, Any, None]
LaplacianRowIterator = Generator[LaplacianRow, Any, None]
SpectralFilterFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

def _iterate_points(geo: hou.Geometry) -> HouPointIterator:
    return geo.iterPoints()

def _iterate_laplacian_rows(geo: hou.Geometry) -> LaplacianRowIterator:
    for point in _iterate_points(geo):
        mass: float = point.attribValue("laplacian_mass")
        column_indices: list[int] = point.attribValue("laplacian_column_indices")
        column_values: list[float] = point.attribValue("laplacian_column_values")
        row = LaplacianRow(
            mass,
            column_indices,
            column_values
        )
        yield row

def _make_laplacian_matrix(
    N: int,
    row_iterator: LaplacianRowIterator
):
    L = lil_matrix((N, N))
    M = np.zeros(N)
    for i, row in enumerate(row_iterator):
        M[i] = row.mass
        for (j, value) in zip(row.column_indices, row.column_values):
            L[i, j] = value * -1
    return (L, diags(M))

def _solve_eigenvalue_problem(
    L: lil_matrix,
    M: dia_matrix,
    k: int = 1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    W, V = eigsh(L.tocsr(), k=k, M=M, sigma=0)
    return W, V.T

def _identity_filter(w: npt.NDArray[np.float64]):
    return np.ones(len(w))

def _make_spectral_transform_matrix(
    V: npt.NDArray[np.float64],
    M: npt.NDArray[np.float64],
    tolerance: float
) -> lil_matrix:
    A = M @ V.T # (N,k)
    mask = np.abs(A) <= tolerance
    A[mask] = 0.0
    return lil_matrix(A)

def _make_spectral_filter_matrix(
    W: npt.NDArray[np.float64],
    V: npt.NDArray[np.float64],
    M: npt.NDArray[np.float64],
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

def _hip_path() -> Path:
    hip_path: str = hou.expandString("$HIP")
    return Path(hip_path)

def _profile_function(function: Callable):
    def profile_enabled() -> bool:
        value = os.getenv("MH_PROFILE")
        if value:
            return value == "1"
        else:
            return False

    def run_profile(*args, **kwargs):
        profiler = cProfile.Profile()

        profiler.enable()
        function(*args, **kwargs)
        profiler.disable()

        file_name = f"python-profile-{function.__name__}.bin"
        file_path = _hip_path().joinpath(file_name)
        profiler.dump_stats(str(file_path))

    if profile_enabled():
        return run_profile
    else:
        return function

@_profile_function
def eigenvalue_eigenvector():
    node: hou.Node = hou.pwd()
    params = NodeParameters(node)

    geo: hou.Geometry = node.geometry()
    geo.addAttrib(hou.attribType.Point, "eigenvalue", 0.0)
    geo.addArrayAttrib(hou.attribType.Point, "eigenvector", hou.attribData.Float)
    geo.addArrayAttrib(hou.attribType.Point, "MHT_indices", hou.attribData.Int)
    geo.addArrayAttrib(hou.attribType.Point, "MHT_values", hou.attribData.Float)

    N: int = geo.pointCount()
    L, M = _make_laplacian_matrix(N, _iterate_laplacian_rows(geo))
    W, V = _solve_eigenvalue_problem(L, M, params.eigenvalue_num)
    A = _make_spectral_transform_matrix(V, M, params.tolerance)
    A_ = A.T

    geo.deletePoints(geo.points())

    for i in range(len(W)):
        point: hou.Point = geo.createPoint()
        point.setAttribValue("eigenvalue", W[i])
        point.setAttribValue("eigenvector", V[i])
        point.setAttribValue("MHT_indices", A_.rows[i])
        point.setAttribValue("MHT_values", A_.data[i])

@_profile_function
def spectral_transform_matrix():
    node: hou.Node = hou.pwd()
    params = NodeParameters(node)

    geo: hou.Geometry = node.geometry()
    geo.addArrayAttrib(hou.attribType.Point, "MHT_indices", hou.attribData.Int)
    geo.addArrayAttrib(hou.attribType.Point, "MHT_values", hou.attribData.Float)

    N: int = geo.pointCount()
    L, M = _make_laplacian_matrix(N, _iterate_laplacian_rows(geo))
    _, V = _solve_eigenvalue_problem(L, M, params.eigenvalue_num)
    A = _make_spectral_transform_matrix(V, M, params.tolerance)

    for i, point in enumerate(_iterate_points(geo)):
        point.setAttribValue("MHT_indices", A.rows[i])
        point.setAttribValue("MHT_values", A.data[i])

@_profile_function
def spectral_filter_matrix(spectral_filter: Optional[SpectralFilterFunction] = None):
    node: hou.Node = hou.pwd()
    params = NodeParameters(node)

    geo: hou.Geometry = node.geometry()
    geo.addArrayAttrib(hou.attribType.Point, "MHT_indices", hou.attribData.Int)
    geo.addArrayAttrib(hou.attribType.Point, "MHT_values", hou.attribData.Float)

    if spectral_filter is None:
        spectral_filter = _identity_filter

    N: int = geo.pointCount()
    L, M = _make_laplacian_matrix(N, _iterate_laplacian_rows(geo))
    W, V = _solve_eigenvalue_problem(L, M, params.eigenvalue_num)
    A = _make_spectral_filter_matrix(W, V, M, params.tolerance, spectral_filter)

    for i, point in enumerate(_iterate_points(geo)):
        point.setAttribValue("MHT_indices", A.rows[i])
        point.setAttribValue("MHT_values", A.data[i])
