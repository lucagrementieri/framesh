from typing import Optional

import numpy as np
import numpy.typing as npt
import trimesh

from .util import get_nearby_indices, timeit


@timeit
def flare_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: Optional[float] = None,
    use_vertex_normal: bool = False,
    *,
    z_radius: Optional[float] = None,
) -> npt.NDArray[np.float64]:
    """
    Reference: A Repeatable and Efficient Canonical Reference for Surface Matching. (3DIMPVT 2012)
    Authors: Petrelli Alioscia, and Luigi Di Stefano
    """
    vertex = mesh.vertices[vertex_index]
    if not use_vertex_normal:
        z_neighbors = get_nearby_indices(mesh, vertex_index, z_radius)
        _, z_axis = trimesh.points.plane_fit(mesh.vertices[z_neighbors])
        if np.dot(z_axis, mesh.vertex_normals[vertex_index]) < 0.0:
            z_axis *= -1
    else:
        z_neighbors = None
        z_axis = mesh.vertex_normals[vertex_index]
    if z_neighbors is not None and radius == z_radius:
        x_neighbors = z_neighbors
    else:
        x_neighbors = get_nearby_indices(mesh, vertex_index, radius)
    distances = trimesh.util.row_norm(mesh.vertices[x_neighbors] - vertex)
    EXCLUDE_RADIUS_COEFFICIENT = 0.85
    exclude_radius = EXCLUDE_RADIUS_COEFFICIENT * (np.max(distances) if radius is None else radius)
    x_neighbors = x_neighbors[distances > exclude_radius]
    x_point_index = np.argmax(np.dot(mesh.vertices[x_neighbors] - vertex, z_axis))
    x_vector = mesh.vertices[x_point_index] - vertex
    y_axis = trimesh.transformations.unit_vector(np.cross(z_axis, x_vector))
    x_axis = np.cross(y_axis, z_axis)
    axes = np.column_stack((x_axis, y_axis, z_axis))
    return axes
