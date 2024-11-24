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
    """Computes a Local Reference Frame (LRF) for a vertex using the FLARE method.

    This function implements the FLARE (Fast Local Axis Reference Extraction) method for computing
    a robust local coordinate system at a given vertex. It uses plane fitting for z-axis computation
    and a distance-based point selection strategy for x-axis determination.

    Args:
        mesh: The input 3D mesh.
        vertex_index: Index of the vertex for which to compute the LRF.
        radius: Support radius for the LRF computation. If None,
            uses the maximum distance from the vertex to any other vertex.
        use_vertex_normal: If True, uses the vertex normal directly as the
            z-axis of the LRF. If False, computes the z-axis from plane fitting.
        z_radius: Support radius for z-axis computation. If None,
            uses the same value as radius.

    Returns:
        A 3x3 matrix where each column represents an axis of the LRF.
        The columns are [x-axis, y-axis, z-axis] forming a right-handed coordinate system.

    Note:
        The implementation follows these steps:
        1. Computes z-axis by plane fitting or using vertex normal
        2. Selects points outside 85% of support radius
        3. Finds point with maximum projection on z-axis
        4. Uses this point to define x-axis direction
        5. Completes right-handed coordinate system

    Reference:
        Petrelli, A., & Di Stefano, L. (2012).
        "A Repeatable and Efficient Canonical Reference for Surface Matching."
        International Conference on 3D Imaging, Modeling, Processing,
        Visualization and Transmission (3DIMPVT).
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
