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
    """Computes a Local Reference Frame (LRF) for a vertex using the SHOT method.

    This function implements the Local Reference Frame computation from the SHOT
    (Signature of Histograms of OrienTations) descriptor. It creates a robust and
    repeatable local coordinate system at a given vertex using weighted covariance
    analysis of neighboring points.

    Args:
        mesh: The input 3D mesh.
        vertex_index: Index of the vertex for which to compute the LRF.
        radius: Support radius for the LRF computation. If None,
            uses the maximum distance from the vertex to any other vertex.
        use_vertex_normal: If True, uses the vertex normal directly as the
            z-axis of the LRF. If False, computes the z-axis from covariance analysis.

    Returns:
        A 3x3 matrix where each column represents an axis of the LRF.
        The columns are [x-axis, y-axis, z-axis] forming a right-handed coordinate system.

    Note:
        The implementation follows these steps:
        1. Identifies neighboring points within the support radius
        2. Computes weighted covariance using distance-based weights
        3. Performs eigendecomposition to get initial axes
        4. Ensures consistent orientation using majority voting and vertex normal
        5. Returns orthonormal axes forming a right-handed coordinate system

    Reference:
        Tombari, F., Salti, S., & Di Stefano, L. (2010).
        "Unique signatures of histograms for local surface description."
        European Conference on Computer Vision (ECCV).
    """
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
