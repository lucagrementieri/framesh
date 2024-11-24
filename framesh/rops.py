from typing import Optional

import numpy as np
import numpy.typing as npt
import trimesh

from .util import timeit


@timeit
def rops_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: Optional[float] = None,
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """Computes the Local Reference Frame (LRF) for a vertex using Rotational Projection Statistics.

    This function implements the LRF computation method described in the paper
    "A local feature descriptor for 3D rigid objects based on rotational projection statistics"
    (ICCSPA 2013). The LRF provides a robust coordinate system for local feature description.

    Args:
        mesh: The input 3D mesh.
        vertex_index: Index of the vertex for which to compute the LRF.
        radius: Support radius for the LRF computation. If None,
            uses the maximum distance from the vertex to any other vertex.
        use_vertex_normal: If True, uses the vertex normal directly as the
            z-axis of the LRF. If False, computes the z-axis from scatter matrix analysis.

    Returns:
        A 3x3 matrix where each column represents an axis of the LRF.
        The columns are [x-axis, y-axis, z-axis] forming a right-handed coordinate system.

    Note:
        The implementation follows these steps:
        1. Computes a weighted scatter matrix using face areas and distances
        2. Performs eigendecomposition to get initial axes
        3. Ensures consistent orientation using vertex normal and projection signs
        4. Returns orthonormal axes forming a right-handed coordinate system

    Reference:
        Guo, Y., Sohel, F. A., Bennamoun, M., Wan, J., & Lu, M. (2013).
        "A local feature descriptor for 3D rigid objects based on rotational projection statistics."
        International Conference on Communications, Signal Processing, and their Applications
        (ICCSPA).
    """
    differences = mesh.vertices - mesh.vertices[vertex_index]
    distances = trimesh.util.row_norm(differences)
    if radius is None:
        radius = np.max(distances)
    d1 = np.expand_dims(differences[mesh.faces], axis=(2, 4))
    d2 = np.expand_dims(differences[mesh.faces], axis=(1, 3))
    dw = np.expand_dims(np.eye(3) + 1, (0, 3, 4))
    face_scatter = np.sum(dw * d1 * d2, axis=(1, 2)) / 12
    area_weights = mesh.area_faces / mesh.area
    face_centers = np.mean(mesh.vertices[mesh.faces], axis=1)
    centers_differences = face_centers - mesh.vertices[vertex_index]
    distance_weights = np.square(radius - trimesh.util.row_norm(centers_differences))
    mesh_scatter = np.sum(
        face_scatter * np.expand_dims(area_weights * distance_weights, axis=(1, 2)),
        axis=0,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(mesh_scatter)
    assert eigenvalues[0] <= eigenvalues[1] <= eigenvalues[2]
    axes = np.fliplr(eigenvectors)
    hx = np.sum(centers_differences.dot(axes[:, 0]) * area_weights * distance_weights)
    if hx < 0:
        axes[:, 0] *= -1
    if use_vertex_normal:
        axes[:, 2] = mesh.vertex_normals[vertex_index]
        axes[:, 1] = np.cross(axes[:, 2], axes[:, 0])
        axes[:, 0] = np.cross(axes[:, 1], axes[:, 2])
    else:
        if np.dot(mesh.vertex_normals[vertex_index], axes[:, 2]) < 0.0:
            axes[:, 2] *= -1
        axes[:, 1] = np.cross(axes[:, 2], axes[:, 0])
    return axes
