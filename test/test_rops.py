import numpy as np
import pytest

import framesh.rops


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_rops_lrf(lrf_input: str, request: pytest.FixtureRequest) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    axes = framesh.rops.rops_lrf(mesh, vertex_index, radius=None)
    assert np.allclose(np.dot(axes.T, axes), np.eye(3))
    assert np.isclose(np.linalg.det(axes), 1.0)

    axes_with_normal = framesh.rops.rops_lrf(mesh, vertex_index, radius=2.0, use_vertex_normal=True)
    assert np.allclose(np.dot(axes_with_normal.T, axes_with_normal), np.eye(3))
    assert np.isclose(np.linalg.det(axes_with_normal), 1.0)
    assert np.allclose(axes_with_normal[:, 2], mesh.vertex_normals[vertex_index])


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_rops_frames(lrf_input: str, request: pytest.FixtureRequest) -> None:
    name, mesh, _ = request.getfixturevalue(lrf_input)
    test_indices = np.array([0, 5, 10, 15])
    # test_indices = (37, 44)

    # Compare with individual LRF computations
    frames = framesh.rops.rops_frames(mesh, test_indices, radius=None)
    for frame, vertex_index in zip(frames, test_indices, strict=True):
        single_frame = framesh.rops.rops_lrf(mesh, vertex_index, radius=None)
        assert np.allclose(frame, single_frame)

    # Compare with individual LRF computations using vertex normals
    frames_with_normal = framesh.rops.rops_frames(
        mesh, test_indices, radius=2.0, use_vertex_normal=True
    )
    for frame, vertex_index in zip(frames_with_normal, test_indices, strict=True):
        single_frame = framesh.rops.rops_lrf(mesh, vertex_index, radius=2.0, use_vertex_normal=True)
        assert np.allclose(frame, single_frame)


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_rops_frames_iterative(lrf_input: str, request: pytest.FixtureRequest) -> None:
    name, mesh, _ = request.getfixturevalue(lrf_input)
    test_indices = (37, 44)

    # Compare with individual LRF computations
    frames = framesh.rops.rops_frames_iterative(mesh, test_indices, radius=None)
    for frame, vertex_index in zip(frames, test_indices, strict=True):
        single_frame = framesh.rops.rops_lrf(mesh, vertex_index, radius=None)
        assert np.allclose(frame, single_frame), f"vertex_index {vertex_index}"

    # Compare with individual LRF computations using vertex normals
    frames_with_normal = framesh.rops.rops_frames_iterative(
        mesh, test_indices, radius=2.0, use_vertex_normal=True
    )
    for frame, vertex_index in zip(frames_with_normal, test_indices, strict=True):
        single_frame = framesh.rops.rops_lrf(mesh, vertex_index, radius=2.0, use_vertex_normal=True)
        assert np.allclose(frame, single_frame)
