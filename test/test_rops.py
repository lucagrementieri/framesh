import numpy as np
import pytest

import framesh.rops


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_rops_lrf(lrf_input: str, request: pytest.FixtureRequest) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    axes = framesh.rops.rops_lrf(mesh, vertex_index, radius=2.0)
    assert np.allclose(np.dot(axes.T, axes), np.eye(3))

    axes_with_normal = framesh.rops.rops_lrf(mesh, vertex_index, radius=2.0, use_vertex_normal=True)
    assert np.allclose(np.dot(axes.T, axes), np.eye(3))
    assert np.allclose(axes_with_normal[:, 2], mesh.vertex_normals[vertex_index])
