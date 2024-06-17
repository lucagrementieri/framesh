from typing import Optional

import numpy as np
import numpy.typing as npt
import trimesh.util

from .util import timeit


@timeit
def shot_local_reference_frame(
    points: npt.NDArray[np.float64],
    center: npt.NDArray[np.float64],
    radius: Optional[float] = None,
) -> npt.NDArray[np.float64]:
    difference = points - center
    distances = trimesh.util.row_norm(difference)
    if radius is None:
        radius = np.max(distances)
    scale_factors = np.maximum(radius - distances, 0.0)
    scale_factors /= scale_factors.sum()
    weighted_covariance = np.einsum("i,ij,ik->jk", scale_factors, difference, difference)
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance)
    assert eigenvalues[0] <= eigenvalues[1]
    assert eigenvalues[1] <= eigenvalues[2]
    axes = np.fliplr(eigenvectors)
    if np.mean(np.dot(difference, axes[:, 0]) >= 0) < 0.5:
        axes[:, 0] *= -1
    if np.mean(np.dot(difference, axes[:, 2]) >= 0) < 0.5:
        axes[:, 2] *= -1
    if np.dot(np.cross(axes[:, 2], axes[:, 0]), axes[1]) < 0:
        axes[:, 1] *= -1
    return axes
