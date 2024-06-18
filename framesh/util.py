import functools
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import trimesh


def timeit(method):
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
    mesh: trimesh.Trimesh, vertex_index: int, radius: Optional[float] = None
) -> npt.NDArray[np.int64]:
    vertex = mesh.vertices[vertex_index]
    if radius is None:
        return np.arange(len(mesh.vertices))
    neighbors = np.array(
        mesh.kdtree.query_ball_point(vertex, radius, workers=-1, return_sorted=False)
    )
    return neighbors
