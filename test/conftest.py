import numpy as np
import pytest
import trimesh


@pytest.fixture(scope="session")
def half_cylinder() -> trimesh.Trimesh:
    origin = np.array([1.0, 1.0, 1.0])
    cylinder = trimesh.creation.cylinder(
        radius=5, segment=np.stack((origin, origin + np.array([10.0, 0.0, 0.0])))
    )
    half_cylinder = cylinder.slice_plane(origin, np.array([0.0, 0.0, 1.0]))
    half_cylinder.export("half_cylinder.ply")
    return half_cylinder


@pytest.fixture(scope="session")
def half_sphere() -> trimesh.Trimesh:
    origin = np.array([2.0, -2.0, 1.0])
    sphere = trimesh.creation.icosphere(radius=5, subdivision=2)
    half_sphere = sphere.slice_plane(origin, np.array([0.0, 0.0, 1.0]))
    half_sphere.export("half_sphere.ply")
    return half_sphere
