from typing import Tuple

import numpy as np
import pytest
import trimesh

import framesh.board
from framesh.util import export_lrf


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_shot_lrf(lrf_input: Tuple[str, trimesh.Trimesh, int], request) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    axes = framesh.board.board_lrf(mesh, vertex_index, radius=3.0, z_radius=2.0)
    assert np.allclose(np.dot(axes.T, axes), np.eye(3))
    export_lrf(f"board_{name}_classic.ply", mesh.vertices[vertex_index], axes)

    axes_with_normal = framesh.board.board_lrf(
        mesh, vertex_index, radius=2.0, use_vertex_normal=True, z_radius=1.0
    )
    assert np.allclose(np.dot(axes.T, axes), np.eye(3))
    assert np.allclose(axes_with_normal[:, 2], mesh.vertex_normals[vertex_index])
    export_lrf(f"board_{name}_normal.ply", mesh.vertices[vertex_index], axes_with_normal)
