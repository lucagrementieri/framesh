import numpy as np
import numpy.typing as npt
import trimesh

from .util import get_nearby_indices, round_zeros, timeit


@timeit
def toldi_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: float | None = None,
    *,
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """Computes the Local Reference Frame (LRF) for a vertex using the TOLDI method.

    This function implements the LRF computation from the TOLDI (Triangular-based Overlapping
    Local Depth Images) descriptor. It creates a robust local coordinate system at a given
    vertex using a combination of PCA and projection-based weighting.

    Args:
        mesh: The input 3D mesh.
        vertex_index: Index of the vertex for which to compute the LRF.
        radius: Support radius for the LRF computation. If None,
            uses the maximum distance from the vertex to any other vertex.
        use_vertex_normal: If True, uses the vertex normal directly as the
            z-axis of the LRF. If False, computes the z-axis from PCA.

    Returns:
        A 3x3 matrix where each column represents an axis of the LRF.
        The columns are [x-axis, y-axis, z-axis] forming a right-handed coordinate system.

    Note:
        The implementation follows these steps:
        1. Computes z-axis using PCA on a smaller neighborhood (radius/3)
        2. Ensures consistent z-axis orientation using vertex normal
        3. Computes x-axis using weighted projections in full neighborhood
        4. Derives y-axis to complete right-handed coordinate system

    Reference:
        Yang, J., Zhang, Q., Xiao, Y., & Cao, Z. (2017).
        "TOLDI: An effective and robust approach for 3D local shape description."
        Pattern Recognition, 65, 175-187.
    """
    vertex = mesh.vertices[vertex_index]
    neighbor_indices = get_nearby_indices(mesh, vertex_index, radius)
    neighbor_vertices = mesh.vertices[neighbor_indices]
    differences = neighbor_vertices - vertex

    if not use_vertex_normal:
        if radius is None:
            z_vertices = neighbor_vertices
        else:
            z_radius = radius / 3.0
            z_neighbors = get_nearby_indices(mesh, vertex_index, z_radius)
            z_vertices = mesh.vertices[z_neighbors]
        z_centroid = np.mean(z_vertices, axis=0)
        centroid_differences = z_vertices - z_centroid
        covariance = np.dot(centroid_differences.T, centroid_differences)
        _, eigenvectors = np.linalg.eigh(covariance)
        z_axis = round_zeros(eigenvectors[:, 0])
        if np.dot(np.sum(differences, axis=0), z_axis) > 0.0:
            z_axis *= -1
    else:
        z_axis = round_zeros(mesh.vertex_normals[vertex_index])
    distances = trimesh.util.row_norm(differences)
    projection_distances = np.dot(differences, z_axis)
    scale_factors = np.square((radius - distances) * projection_distances)
    x_axis = round_zeros(np.dot(differences.T, scale_factors))
    y_axis = trimesh.transformations.unit_vector(np.cross(z_axis, x_axis))
    if np.isnan(y_axis).any():
        x_axis = np.roll(z_axis, 1)
        y_axis = np.cross(z_axis, x_axis)
    else:
        x_axis = np.cross(y_axis, z_axis)
    axes = np.column_stack((x_axis, y_axis, z_axis))
    return axes
