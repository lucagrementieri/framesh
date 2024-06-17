from typing import Optional

import numpy as np
import numpy.typing as npt
import trimesh

from .util import highlight_vertices, timeit


@timeit
def board_lrf(
    mesh: trimesh.Trimesh,
    vertex_index: int,
    radius: Optional[float] = None,
    use_vertex_normal: bool = False,
    *,
    z_radius: Optional[float] = None,
) -> npt.NDArray[np.float64]:
    """
    Reference: On the repeatability of the local reference frame for partial shape matching. (ICCV 2011)
    Authors: Petrelli Alioscia, and Luigi Di Stefano
    """
    vertex = mesh.vertices[vertex_index]
    if not use_vertex_normal:
        z_neighbors = mesh.kdtree.query_ball_point(
            vertex,
            np.inf if z_radius is None else z_radius,
            workers=-1,
            return_sorted=False,
        )
        highlight_vertices("boardz.ply", mesh, z_neighbors)
        _, z_axis = trimesh.points.plane_fit(mesh.vertices[z_neighbors])
        if np.dot(z_axis, mesh.vertex_normals[vertex_index]) < 0.0:
            z_axis *= -1
    else:
        z_neighbors = None
        z_axis = mesh.vertex_normals[vertex_index]
    if z_neighbors is not None and radius == z_radius:
        x_neighbors = z_neighbors
    else:
        x_neighbors = mesh.kdtree.query_ball_point(
            vertex,
            np.inf if radius is None else radius,
            workers=-1,
            return_sorted=False,
        )
    x_point_index = np.argmin(np.abs(np.dot(mesh.vertex_normals[x_neighbors], z_axis)))
    x_vector = mesh.vertices[x_point_index] - vertex
    x_axis = trimesh.unitize(x_vector - np.dot(x_vector, z_axis) * z_axis)
    y_axis = np.cross(z_axis, x_axis)
    axes = np.column_stack((x_axis, y_axis, z_axis))
    return axes
