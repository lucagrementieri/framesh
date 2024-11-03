from typing import Tuple

import numpy as np
import pytest
import trimesh

import framesh.gframes
from framesh.util import export_lrf


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_gframes_lrf(lrf_input: Tuple[str, trimesh.Trimesh, int], request) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    for field_name in ("mean_curvature", "gaussian_curvature", "fiedler_squared"):
        field = getattr(framesh.gframes, field_name)(mesh)
        axes = framesh.gframes.gframes_lrf(mesh, vertex_index, radius=3.0, scalar_field=field)
        assert np.allclose(np.dot(axes.T, axes), np.eye(3))
        export_lrf(f"gframes_{name}_{field_name}_classic.ply", mesh.vertices[vertex_index], axes)

        axes_with_normal = framesh.gframes.gframes_lrf(
            mesh, vertex_index, radius=3.0, use_vertex_normal=True, scalar_field=field
        )
        assert np.allclose(np.dot(axes_with_normal.T, axes_with_normal), np.eye(3))
        assert np.allclose(axes_with_normal[:, 2], mesh.vertex_normals[vertex_index])
        export_lrf(
            f"gframes_{name}_{field_name}_normal.ply", mesh.vertices[vertex_index], axes_with_normal
        )
