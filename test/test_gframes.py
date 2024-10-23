from typing import Tuple

import numpy as np
import pytest
import trimesh

import framesh.gframes
from framesh.util import export_lrf


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_laplacian(lrf_input: Tuple[str, trimesh.Trimesh, int], request) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    # framesh.gframes.fiedler_squared(mesh)


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_gframes_lrf(lrf_input: Tuple[str, trimesh.Trimesh, int], request) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    print(framesh.gframes.mean_curvature(mesh))
    for field_name in ("mean_curvature", "gaussian_curvature"):
        field = getattr(framesh.gframes, field_name)(mesh)
        axes = framesh.gframes.gframes_lrf(mesh, vertex_index, radius=3.0, scalar_field=field)
        assert np.allclose(np.dot(axes.T, axes), np.eye(3))
        assert np.allclose(axes[:, 2], mesh.vertex_normals[vertex_index])
        export_lrf(f"gframes_{name}_{field_name}.ply", mesh.vertices[vertex_index], axes)
