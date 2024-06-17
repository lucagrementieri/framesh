from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import trimesh


def highlight_vertices(
    output_path: Union[str, Path],
    mesh: trimesh.Trimesh,
    vertex_indices: npt.NDArray[np.int64],
    color: npt.NDArray[np.float64] = np.array([1.0, 0.0, 0.0]),
    radius: float = 0.1,
) -> None:
    color_mesh = mesh.copy()
    color_mesh.visual.vertex_colors = np.full(3, 0.5)
    meshes = [color_mesh]
    for vertex_index in vertex_indices:
        vertex_sphere = trimesh.creation.icosphere(radius=radius, vertex_colors=color)
        vertex_sphere.apply_translation(mesh.vertices[vertex_index])
        meshes.append(vertex_sphere)
    export_mesh = trimesh.util.concatenate(meshes)
    export_mesh.export(output_path)
