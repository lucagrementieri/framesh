import numpy as np
import pytest

import framesh.flare


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_flare_lrf(lrf_input: str, request: pytest.FixtureRequest) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    axes = framesh.flare.flare_lrf(mesh, vertex_index, radius=3.0, z_radius=2.0)
    assert np.allclose(np.dot(axes.T, axes), np.eye(3))
    assert np.isclose(np.linalg.det(axes), 1.0)

    axes_with_normal = framesh.flare.flare_lrf(
        mesh, vertex_index, radius=2.0, use_vertex_normal=True, z_radius=1.0
    )
    assert np.allclose(np.dot(axes_with_normal.T, axes_with_normal), np.eye(3))
    assert np.isclose(np.linalg.det(axes_with_normal), 1.0)
    assert np.allclose(axes_with_normal[:, 2], mesh.vertex_normals[vertex_index])
