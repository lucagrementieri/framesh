from typing import Optional

import numpy as np
import numpy.typing as npt
import trimesh

from .util import get_nearby_indices, timeit


@timeit
def toldi_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: Optional[float] = None,
    use_vertex_normal: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Reference: TOLDI: An effective and robust approach for 3D local shape description. (2017)
    Authors: Jiaqi Yang, Qian Zhang, Yang Xiao, and Zhiguo Cao.
    """
    vertex = mesh.vertices[vertex_index]
    if not use_vertex_normal:
        z_neighbors = get_nearby_indices(mesh, vertex_index, radius / 3.0)
        z_vertices = mesh.vertices[z_neighbors]
        z_centroid = np.mean(z_vertices, axis=0)
        differences = z_vertices - z_centroid
        covariance = np.dot(differences.T, differences)
        _, eigenvectors = np.linalg.eigh(covariance)
        z_axis = eigenvectors[:, 0]
        if np.dot(mesh.vertex_normals[vertex_index], z_axis) < 0.0:
            z_axis *= -1
    else:
        z_axis = mesh.vertex_normals[vertex_index]
    x_neighbors = get_nearby_indices(mesh, vertex_index, radius)
    x_vertices = mesh.vertices[x_neighbors]
    differences = x_vertices - vertex
    distances = trimesh.util.row_norm(differences)
    projection_distances = np.dot(differences, z_axis)
    scale_factors = np.square((radius - distances) * projection_distances)
    x_axis = np.dot(x_vertices.T, scale_factors)
    y_axis = trimesh.transformations.unit_vector(np.cross(z_axis, x_axis))
    x_axis = np.cross(y_axis, z_axis)
    axes = np.column_stack((x_axis, y_axis, z_axis))
    return axes
