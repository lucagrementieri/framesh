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
