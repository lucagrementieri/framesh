from typing import Tuple

import numpy as np
import pytest
import trimesh

import framesh.gframes
from framesh.util import export_lrf


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_laplacian(lrf_input: Tuple[str, trimesh.Trimesh, int], request) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    framesh.gframes.fiedler_squared(mesh)


def test_curvature() -> None:
    # cylinder = trimesh.creation.cylinder(radius=5.0, height=10.0)
    # cylinder = cylinder.subdivide().subdivide().subdivide()
    # framesh.gframes.curvature(cylinder)

    cylinder = trimesh.creation.cylinder(radius=5.0, height=10.0)
    half_cylinder: trimesh.Trimesh = cylinder.slice_plane(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    half_cylinder = half_cylinder.subdivide().subdivide().subdivide()
    framesh.gframes.curvature(half_cylinder)
