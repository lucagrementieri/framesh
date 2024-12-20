import functools
import time
from collections.abc import Callable
from pathlib import Path

import networkx as nx
import numpy as np
import numpy.typing as npt
import trimesh
from scipy.spatial import KDTree

DEFAULT_COLORS = np.eye(3)
ABSOLUTE_TOLERANCE = 1e-12


def round_zeros(x: npt.NDArray, atol: float = ABSOLUTE_TOLERANCE) -> npt.NDArray:
    """Rounds values close to zero to exactly zero.

    Args:
        x: Input array to process.
        atol: Absolute tolerance for considering a value to be zero.
            Default is ABSOLUTE_TOLERANCE.

    Returns:
        Array with values close to zero (within atol) set to exactly zero,
        and other values unchanged.
    """
    return np.where(np.isclose(x, 0.0, rtol=0, atol=atol), 0, x)


def robust_sign(x: npt.NDArray, atol: float = ABSOLUTE_TOLERANCE) -> npt.NDArray:
    """Returns a robust sign array that handles values close to zero.

    Computes the sign (-1, 0, or 1) for each element in the input array,
    treating values within the absolute tolerance of zero as 0.

    Args:
        x: Input array to compute signs for.
        atol: Absolute tolerance for considering a value to be zero.
            Default is ABSOLUTE_TOLERANCE.

    Returns:
        Array of signs (-1, 0, or 1) with the same shape as the input,
        where values within atol of zero get sign 0.
    """
    sign_array = np.zeros_like(x)
    np.sign(x, out=sign_array, where=np.logical_not(np.isclose(x, 0, rtol=0.0, atol=atol)))
    return sign_array


def timeit(method: Callable) -> Callable:
    """Decorator that prints the execution time of a method.

    Args:
        method: The method to time.

    Returns:
        A wrapped version of the method that prints its execution time.
    """

    @functools.wraps(method)
    def timed(*args, **kw):  # noqa: ANN002, ANN003, ANN202
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print(f"{method.__name__} took {((te - ts) * 1000):2.2f} ms")
        return result

    return timed


def highlight_vertices(
    output_path: str | Path,
    mesh: trimesh.Trimesh,
    vertex_indices: npt.NDArray[np.int64],
    color: npt.NDArray[np.float64] = DEFAULT_COLORS[0],
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
    output_path: str | Path,
    center: npt.NDArray[np.float64],
    lrf: npt.NDArray[np.float64],
    colors: npt.NDArray[np.float64] = DEFAULT_COLORS,
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
    for axis, color in zip(lrf.T, colors, strict=True):
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
    vertex_indices: int | npt.NDArray[np.int_],
    radius: float | npt.NDArray[np.float64],
    *,
    exclude_self: bool = False,
) -> npt.NDArray[np.int64] | list[npt.NDArray[np.int64]]:
    """Gets indices of vertices within a specified radius of target vertices.

    Args:
        mesh: The input mesh.
        vertex_indices: Index or array of indices of target vertices.
        radius: Maximum distance(s) from target vertices. Can be a single float or an array
            matching vertex_index length.
        exclude_self: Whether to exclude the target vertices from the results.

    Returns:
        If vertex_indices is an int: Array of vertex indices within radius of the target vertex.
        If vertex_indices is an array: List of arrays containing vertex indices within radius
            of each target vertex.
    """
    center_vertices = mesh.vertices[vertex_indices]
    row_indices, _ = trimesh.grouping.unique_rows(mesh.vertices)
    unique_vertices = mesh.vertices[row_indices]
    kdtree = KDTree(unique_vertices)
    neighbors = kdtree.query_ball_point(center_vertices, radius, workers=-1, return_sorted=True)
    if exclude_self:
        d, self_indices = kdtree.query(
            center_vertices, distance_upper_bound=ABSOLUTE_TOLERANCE, workers=-1
        )
        assert np.allclose(d, 0.0)
    mesh_neighbors: npt.NDArray[np.int64] | list[npt.NDArray[np.int64]]
    if center_vertices.ndim == 1:
        mesh_neighbors = np.array(neighbors)
        if exclude_self:
            mesh_neighbors = np.delete(
                mesh_neighbors, np.searchsorted(mesh_neighbors, self_indices)
            )
        mesh_neighbors = np.sort(row_indices[mesh_neighbors])
    else:
        assert not isinstance(vertex_indices, int)
        mesh_neighbors = [np.array(n) for n in neighbors]
        if exclude_self:
            assert isinstance(self_indices, np.ndarray)
            mesh_neighbors = [
                np.delete(n, np.searchsorted(n, self_index))
                for n, self_index in zip(mesh_neighbors, self_indices, strict=True)
            ]
        mesh_neighbors = [np.sort(row_indices[n]) for n in mesh_neighbors]
    return mesh_neighbors


def get_nearby_face_indices(
    mesh: trimesh.Trimesh,
    vertex_indices: int | npt.NDArray[np.int_],
    radius: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.int64] | list[npt.NDArray[np.int64]]:
    """Gets indices of faces within a specified radius of target vertices.

    Args:
        mesh: The input mesh.
        vertex_indices: Index or array of indices of target vertices.
        radius: Maximum distance(s) from target vertices. Can be a single float or an array
            matching vertex_index length.
        exclude_self: Whether to exclude the target vertices from the results.

    Returns:
        If vertex_indices is an int: Array of face indices within radius of the target vertex.
        If vertex_indices is an array: List of arrays containing face indices within radius
            of each target vertex.
    """
    center_vertices = mesh.vertices[vertex_indices]
    neighbors = mesh.kdtree.query_ball_point(center_vertices, radius, workers=-1)
    face_indices: npt.NDArray[np.int64] | list[npt.NDArray[np.int64]]
    if center_vertices.ndim == 1:
        face_indices = np.flatnonzero(np.any(np.isin(mesh.faces, neighbors), axis=-1))
    else:
        assert not isinstance(vertex_indices, int)
        face_indices = [
            np.flatnonzero(np.any(np.isin(mesh.faces, vertex_neighbors), axis=-1))
            for vertex_neighbors in neighbors
        ]
    return face_indices


def get_connected_nearby_indices(
    mesh: trimesh.Trimesh,
    vertex_indices: int | npt.NDArray[np.int_],
    radius: float | npt.NDArray[np.float64],
    *,
    exclude_self: bool = False,
) -> npt.NDArray[np.int64] | list[npt.NDArray[np.int64]]:
    """Gets indices of vertices within a specified radius of target vertices and connected to them.

    Args:
        mesh: The input mesh.
        vertex_indices: Index or array of indices of target vertices.
        radius: Maximum distance(s) from target vertices. Can be a single float or an array
            matching vertex_index length.
        exclude_self: Whether to exclude the target vertices from the results.

    Returns:
        If vertex_indices is an int: Array of connected vertex indices within radius of the target
            vertex.
        If vertex_indices is an array: List of arrays containing connected vertex indices within
            radius of each target vertex.
    """
    is_single_index = not hasattr(vertex_indices, "__len__")
    np_vertex_indices = np.array([vertex_indices]) if is_single_index else np.array(vertex_indices)
    center_vertices = mesh.vertices[np_vertex_indices]
    neighbors = mesh.kdtree.query_ball_point(center_vertices, radius, workers=-1)
    connected_indices = []
    for vertex_index, vertex_neighbors in zip(np_vertex_indices, neighbors, strict=True):
        if len(vertex_neighbors) == 1:
            connected_indices.append(
                np.empty(0, dtype=np.int64) if exclude_self else np.array([vertex_index])
            )
            continue
        neighborhood_edges = mesh.edges_unique[
            np.all(np.isin(mesh.edges_unique, vertex_neighbors), axis=1)
        ]
        if np.all(neighborhood_edges != vertex_index):
            connected_indices.append(
                np.empty(0, dtype=np.int64) if exclude_self else np.array([vertex_index])
            )
            continue
        neighborhood_graph = nx.from_edgelist(neighborhood_edges)
        connected_vertex_indices = nx.node_connected_component(
            neighborhood_graph, int(vertex_index)
        )
        if exclude_self:
            connected_vertex_indices.remove(int(vertex_index))
        connected_indices.append(np.array(sorted(connected_vertex_indices)))
    if is_single_index:
        return connected_indices[0]
    return connected_indices


def get_connected_nearby_face_indices(
    mesh: trimesh.Trimesh,
    vertex_indices: int | npt.NDArray[np.int_],
    radius: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.int64] | list[npt.NDArray[np.int64]]:
    """Gets indices of faces within a specified radius of target vertices and connected to them.

    Args:
        mesh: The input mesh.
        vertex_indices: Index or array of indices of target vertices.
        radius: Maximum distance(s) from target vertices. Can be a single float or an array
            matching vertex_index length.

    Returns:
        If vertex_indices is an int: Array of connected face indices within radius of the target
            vertex.
        If vertex_indices is an array: List of arrays containing connected face indices within
            radius of each target vertex.
    """
    is_single_index = not hasattr(vertex_indices, "__len__")
    np_vertex_indices = np.array([vertex_indices]) if is_single_index else np.array(vertex_indices)
    connected_face_indices = [
        np.flatnonzero(np.any(np.isin(mesh.faces, connected_indices), axis=-1))
        for connected_indices in get_connected_nearby_indices(mesh, np_vertex_indices, radius)
    ]
    if is_single_index:
        return connected_face_indices[0]
    return connected_face_indices
