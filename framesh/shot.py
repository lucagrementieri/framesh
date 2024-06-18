from typing import Optional

import numpy as np
import numpy.typing as npt
import trimesh

from .util import get_nearby_indices, timeit


@timeit
def shot_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: Optional[float] = None,
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Reference: Unique signatures of histograms for local surface description. (ECCV 2010)
    Authors: Tombari Federico, Samuele Salti, and Luigi Di Stefano.
    """
    # TODO: use query
    vertex = mesh.vertices[vertex_index]
    if radius is None:
        differences = mesh.vertices - vertex
        distances = trimesh.util.row_norm(differences)
        radius = np.max(distances)
    else:
        neighbors = get_nearby_indices(mesh, vertex_index, radius)
        differences = mesh.vertices[neighbors] - vertex
        distances = trimesh.util.row_norm(differences)
    assert np.all(distances <= radius)
    scale_factors = radius - distances
    scale_factors /= scale_factors.sum()
    weighted_covariance = np.einsum("i,ij,ik->jk", scale_factors, differences, differences)
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance)
    assert eigenvalues[0] <= eigenvalues[1] <= eigenvalues[2]
    axes = np.fliplr(eigenvectors)
    if np.mean(np.dot(differences, axes[:, 0]) >= 0) < 0.5:
        axes[:, 0] *= -1
    if use_vertex_normal:
        axes[:, 2] = mesh.vertex_normals[vertex_index]
        axes[:, 1] = np.cross(axes[:, 2], axes[:, 0])
        axes[:, 0] = np.cross(axes[:, 1], axes[:, 2])
    else:
        if np.dot(mesh.vertex_normals[vertex_index], axes[:, 2]) < 0.0:
            axes[:, 2] *= -1
        if np.dot(np.cross(axes[:, 2], axes[:, 0]), axes[1]) < 0:
            axes[:, 1] *= -1
    return axes
