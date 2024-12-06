import numpy as np
import pytest

import framesh.flare


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_flare_lrf(lrf_input: str, request: pytest.FixtureRequest) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    axes = framesh.flare.flare_lrf(mesh, vertex_index, radius=1.5, z_radius=1.0)
    assert np.allclose(np.dot(axes.T, axes), np.eye(3))
    assert np.isclose(np.linalg.det(axes), 1.0)

    none_z_radius_axes = framesh.flare.flare_lrf(mesh, vertex_index, radius=1.5)
    assert np.allclose(np.dot(none_z_radius_axes.T, none_z_radius_axes), np.eye(3))
    assert np.isclose(np.linalg.det(none_z_radius_axes), 1.0)

    axes_with_normal = framesh.flare.flare_lrf(
        mesh, vertex_index, radius=2.0, use_vertex_normal=True, z_radius=1.0
    )
    assert np.allclose(np.dot(axes_with_normal.T, axes_with_normal), np.eye(3))
    assert np.isclose(np.linalg.det(axes_with_normal), 1.0)
    assert np.allclose(axes_with_normal[:, 2], mesh.vertex_normals[vertex_index])


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_flare_frames(lrf_input: str, request: pytest.FixtureRequest) -> None:
    name, mesh, _ = request.getfixturevalue(lrf_input)
    test_indices = np.array([0, 5, 10, 15])

    # Compare with individual LRF computations
    frames = framesh.flare.flare_frames(mesh, test_indices, radius=1.5, z_radius=1.0)
    for frame, vertex_index in zip(frames, test_indices, strict=True):
        single_frame = framesh.flare.flare_lrf(mesh, vertex_index, radius=1.5, z_radius=1.0)
        assert np.allclose(frame, single_frame)

    none_z_radius_frames = framesh.flare.flare_frames(mesh, test_indices, radius=1.5)
    for frame, vertex_index in zip(none_z_radius_frames, test_indices, strict=True):
        single_frame = framesh.flare.flare_lrf(mesh, vertex_index, radius=1.5)
        assert np.allclose(frame, single_frame)

    # Compare with individual LRF computations using vertex normals
    frames_with_normal = framesh.flare.flare_frames(
        mesh, test_indices, radius=2.0, use_vertex_normal=True, z_radius=1.0
    )
    for frame, vertex_index in zip(frames_with_normal, test_indices, strict=True):
        single_frame = framesh.flare.flare_lrf(
            mesh, vertex_index, radius=2.0, use_vertex_normal=True, z_radius=1.0
        )
        assert np.allclose(frame, single_frame)
