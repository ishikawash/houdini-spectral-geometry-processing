from typing import Any, Generator, Callable, Protocol
import numpy as np
import numpy.typing as npt
import hou

class LaplacianRow(Protocol):

    @property
    def mass(self) -> float: ...

    @property
    def column_indices(self) -> list[int]: ...

    @property
    def column_values(self) -> list[float]: ...

LaplacianRowIterator = Generator[LaplacianRow, Any, None]
HouPointIterator = Generator[hou.Point, Any, None]
SpectralFilterFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
