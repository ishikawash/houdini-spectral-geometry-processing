import typing
from typing import Callable, Optional, Protocol, runtime_checkable
import os
import dataclasses
import cProfile
from pathlib import Path
import numpy as np
import numpy.typing as npt
from scipy.sparse import lil_matrix, dia_matrix, diags
import hou
from ._types import LaplacianRow, LaplacianRowIterator, HouPointIterator, SpectralFilterFunction
from . import _functions


@dataclasses.dataclass(frozen=True)
class LaplacianRowData:
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


@runtime_checkable
class Functions(Protocol):

    def make_spectral_transform_matrix(
        self,
        V: npt.NDArray[np.float64],
        M: npt.NDArray[np.float64],
        tolerance: float
    ) -> lil_matrix:    
        ...

    def make_spectral_filter_matrix(
        self,
        W: npt.NDArray[np.float64],
        V: npt.NDArray[np.float64],
        M: npt.NDArray[np.float64],
        tolerance: float,
        filter: SpectralFilterFunction
    ) -> lil_matrix:
        ...

    def solve_eigenvalue_problem(
        self,
        L: lil_matrix,
        M: dia_matrix,
        k: int = 1,
        tolerance: float = 1e-6
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        ...


def _iterate_points(geo: hou.Geometry) -> HouPointIterator:
    return geo.iterPoints()

def _iterate_laplacian_rows(geo: hou.Geometry) -> LaplacianRowIterator:
    for point in _iterate_points(geo):
        mass: float = point.attribValue("laplacian_mass")
        column_indices: list[int] = point.attribValue("laplacian_column_indices")
        column_values: list[float] = point.attribValue("laplacian_column_values")
        row = LaplacianRowData(
            mass,
            column_indices,
            column_values
        )
        yield typing.cast(LaplacianRow, row) 

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

def _identity_filter(w: npt.NDArray[np.float64]):
    return np.ones(len(w))

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

def _get_functions() -> Functions:
    def cupy_enabled() -> bool:
        value = os.getenv("MH_ENABLE_CUPY")
        return value == "1"

    if cupy_enabled():
        return _functions.cupy
    else:
        return _functions.default


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
    funcs: Functions = _get_functions()
    W, V = funcs.solve_eigenvalue_problem(L, M, params.eigenvalue_num)
    A = funcs.make_spectral_transform_matrix(V, M, params.tolerance)
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
    funcs: Functions = _get_functions()
    _, V = funcs.solve_eigenvalue_problem(L, M, params.eigenvalue_num)
    A = funcs.make_spectral_transform_matrix(V, M, params.tolerance)

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
    funcs: Functions = _get_functions()
    W, V = funcs.solve_eigenvalue_problem(L, M, params.eigenvalue_num)
    A = funcs.make_spectral_filter_matrix(W, V, M, params.tolerance, spectral_filter)

    for i, point in enumerate(_iterate_points(geo)):
        point.setAttribValue("MHT_indices", A.rows[i])
        point.setAttribValue("MHT_values", A.data[i])
