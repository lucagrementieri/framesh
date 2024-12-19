import numpy as np
import pytest

import framesh.gframes


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_gframes_lrf(lrf_input: str, request: pytest.FixtureRequest) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    vertex_index = 30
    for field_name in ("mean_curvature", "gaussian_curvature", "fiedler_squared"):
        field = getattr(framesh.gframes, field_name)(mesh)
        axes = framesh.gframes.gframes_lrf(mesh, vertex_index, radius=1.0, scalar_field=field)
        assert np.allclose(np.dot(axes.T, axes), np.eye(3))
        assert np.isclose(np.linalg.det(axes), 1.0)

        axes_with_normal = framesh.gframes.gframes_lrf(
            mesh, vertex_index, radius=1.5, use_vertex_normal=True, scalar_field=field
        )
        assert np.allclose(np.dot(axes_with_normal.T, axes_with_normal), np.eye(3))
        assert np.isclose(np.linalg.det(axes_with_normal), 1.0)
        assert np.allclose(axes_with_normal[:, 2], mesh.vertex_normals[vertex_index])


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_gframes_frames(lrf_input: str, request: pytest.FixtureRequest) -> None:
    name, mesh, _ = request.getfixturevalue(lrf_input)
    test_indices = np.array([5, 10, 15])

    for field_name in ("mean_curvature", "gaussian_curvature", "fiedler_squared"):
        field = getattr(framesh.gframes, field_name)(mesh)
        # Compare with individual LRF computations
        frames = framesh.gframes.gframes_frames(mesh, test_indices, radius=1.0, scalar_field=field)
        for frame, vertex_index in zip(frames, test_indices, strict=True):
            single_frame = framesh.gframes.gframes_frames(
                mesh, vertex_index, radius=1.0, scalar_field=field
            )
            assert np.allclose(frame, single_frame)

        # Compare with individual LRF computations using vertex normals
        frames_with_normal = framesh.gframes.gframes_frames(
            mesh, test_indices, radius=2.0, use_vertex_normal=True, scalar_field=field
        )
        for frame, vertex_index in zip(frames_with_normal, test_indices, strict=True):
            single_frame = framesh.gframes.gframes_frames(
                mesh, vertex_index, radius=2.0, use_vertex_normal=True, scalar_field=field
            )
            assert np.allclose(frame, single_frame)
