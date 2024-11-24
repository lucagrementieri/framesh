
import numpy as np
import pytest
import trimesh


def half_cylinder() -> trimesh.Trimesh:
    cylinder = trimesh.creation.cylinder(radius=5.0, height=10.0)
    half_cylinder: trimesh.Trimesh = cylinder.slice_plane(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    half_cylinder = half_cylinder.subdivide().subdivide().subdivide()
    half_cylinder.export("half_cylinder.ply")
    return half_cylinder


@pytest.fixture(scope="session")
def half_cylinder_right() -> tuple[str, trimesh.Trimesh, int]:
    mesh = half_cylinder()
    _, vertex_index = mesh.kdtree.query(np.array([5.0, 0.0, 0.0]))
    return "half_cylinder", mesh, vertex_index


def half_sphere() -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(radius=5, subdivision=2)
    half_sphere: trimesh.Trimesh = sphere.slice_plane(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    half_sphere.export("half_sphere.ply")
    return half_sphere


@pytest.fixture(scope="session")
def half_sphere_top() -> tuple[str, trimesh.Trimesh, int]:
    mesh = half_sphere()
    vertex_index = int(np.argmax(mesh.vertices[:, 2]))
    return "half_sphere", mesh, vertex_index
