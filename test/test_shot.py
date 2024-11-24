from typing import Tuple

import numpy as np
import pytest
import trimesh

import framesh.shot


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_shot_lrf(lrf_input: Tuple[str, trimesh.Trimesh, int], request) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    axes = framesh.shot.shot_lrf(mesh, vertex_index, radius=None)
    assert np.allclose(np.dot(axes.T, axes), np.eye(3))

    axes_with_normal = framesh.shot.shot_lrf(mesh, vertex_index, radius=2.0, use_vertex_normal=True)
    assert np.allclose(np.dot(axes.T, axes), np.eye(3))
    assert np.allclose(axes_with_normal[:, 2], mesh.vertex_normals[vertex_index])


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_shot_frames(lrf_input: Tuple[str, trimesh.Trimesh, int], request) -> None:
    name, mesh, _ = request.getfixturevalue(lrf_input)
    test_indices = np.array([0, 5, 10, 15])

    # Compare with individual LRF computations
    frames = framesh.shot.shot_frames(mesh, test_indices, radius=None)
    for frame, vertex_index in zip(frames, test_indices):
        single_frame = framesh.shot.shot_lrf(mesh, vertex_index, radius=None)
        assert np.allclose(frame, single_frame)

    # Compare with individual LRF computations using vertex normals
    frames_with_normal = framesh.shot.shot_frames(
        mesh, test_indices, radius=2.0, use_vertex_normal=True
    )
    for frame, vertex_index in zip(frames_with_normal, test_indices):
        single_frame = framesh.shot.shot_lrf(mesh, vertex_index, radius=2.0, use_vertex_normal=True)
        assert np.allclose(frame, single_frame)
