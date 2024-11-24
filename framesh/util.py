import functools
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import trimesh


def timeit(method):
    """Decorator that prints the execution time of a method.

    Args:
        method: The method to time.

    Returns:
        A wrapped version of the method that prints its execution time.
    """

    @functools.wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def highlight_vertices(
    output_path: Union[str, Path],
    mesh: trimesh.Trimesh,
    vertex_indices: npt.NDArray[np.int64],
    color: npt.NDArray[np.float64] = np.array([1.0, 0.0, 0.0]),
    point_radius: float = 0.1,
) -> None:
    """Exports a mesh with highlighted vertices.

    Creates a visualization where specified vertices are marked with colored spheres.

    Args:
        output_path: Path where the output mesh will be saved.
        mesh: The input mesh to visualize.
        vertex_indices: Indices of vertices to highlight.
        color: RGB color values for the highlight spheres.
        point_radius: Radius of the highlight spheres.
    """
    color_mesh = mesh.copy()
    color_mesh.visual.vertex_colors = np.full(3, 0.5)
    meshes = [color_mesh]
    for vertex_index in vertex_indices:
        vertex_sphere = trimesh.creation.icosphere(radius=point_radius, vertex_colors=color)
        vertex_sphere.apply_translation(mesh.vertices[vertex_index])
        meshes.append(vertex_sphere)
    export_mesh = trimesh.util.concatenate(meshes)
    export_mesh.export(output_path)


def export_lrf(
    output_path: Union[str, Path],
    center: npt.NDArray[np.float64],
    lrf: npt.NDArray[np.float64],
    colors: npt.NDArray[np.float64] = np.eye(3),
    axis_radius: float = 0.1,
    axis_length: float = 5.0,
) -> None:
    """Exports a visualization of a Local Reference Frame (LRF).

    Creates a visualization where each axis of the LRF is represented by a colored cylinder.

    Args:
        output_path: Path where the output mesh will be saved.
        center: 3D coordinates of the LRF origin.
        lrf: 3x3 matrix where each column represents an axis of the LRF.
        colors: RGB colors for each axis. Default uses red, green, blue.
        axis_radius: Radius of the cylinder representing each axis.
        axis_length: Length of the cylinder representing each axis.
    """
    markers = []
    for axis, color in zip(lrf.T, colors):
        end_point = center + axis_length * axis
        axis_cylinder = trimesh.creation.cylinder(
            radius=axis_radius,
            segment=np.vstack([center, end_point]),
            vertex_colors=color,
        )
        markers.append(axis_cylinder)
    markers_mesh = trimesh.util.concatenate(markers)
    markers_mesh.export(output_path)


def get_nearby_indices(
    mesh: trimesh.Trimesh,
    vertex_indices: Union[int, npt.NDArray[np.int_]],
    radius: Optional[Union[float, npt.NDArray[np.float64]]] = None,
) -> Union[npt.NDArray[np.int64], list[npt.NDArray[np.int64]]]:
    """Gets indices of vertices within a specified radius of target vertices.

    Args:
        mesh: The input mesh.
        vertex_index: Index or array of indices of target vertices.
        radius: Maximum distance(s) from target vertices. If None, returns all vertices.
            Can be a single float or an array matching vertex_index length.

    Returns:
        If vertex_index is an int: Array of vertex indices within radius of the target vertex.
        If vertex_index is an array: List of arrays containing vertex indices within radius
            of each target vertex.
    """
    center_vertices = mesh.vertices[vertex_indices]
    if radius is None:
        if center_vertices.ndim == 1:
            return np.arange(len(mesh.vertices))
        else:
            return [np.arange(len(mesh.vertices))] * len(center_vertices)
    neighbors = mesh.kdtree.query_ball_point(
        center_vertices, radius, workers=-1, return_sorted=False
    )
    np_neighbors: Union[npt.NDArray[np.int64], list[npt.NDArray[np.int64]]]
    if center_vertices.ndim == 1:
        np_neighbors = np.array(neighbors)
    else:
        np_neighbors = [np.array(n) for n in neighbors]
    return np_neighbors
