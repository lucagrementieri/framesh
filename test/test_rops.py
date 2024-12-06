import numpy as np
import pytest

import framesh.rops


@pytest.mark.parametrize("lrf_input", ["half_cylinder_right", "half_sphere_top"])
def test_rops_lrf(lrf_input: str, request: pytest.FixtureRequest) -> None:
    name, mesh, vertex_index = request.getfixturevalue(lrf_input)
    axes = framesh.rops.rops_lrf(mesh, vertex_index, radius=0.5)
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

    # Compare with individual LRF computations
    frames = framesh.rops.rops_frames(mesh, test_indices, radius=0.5)
    for frame, vertex_index in zip(frames, test_indices, strict=True):
        single_frame = framesh.rops.rops_lrf(mesh, vertex_index, radius=0.5)
        assert np.allclose(frame, single_frame)
    print(frames)
    """
    [[[ 1.          0.         -0.        ]
    [ 0.         -1.         -0.        ]
    [ 0.          0.         -1.        ]]

    [[-0.98077899  0.00588316 -0.19503322]
    [-0.19494202 -0.07245838  0.97813465]
    [-0.00837727  0.99735409  0.07221254]]

    [[-0.70697998 -0.05697844  0.70493459]
    [-0.70718395  0.04514739 -0.70558669]
    [ 0.00837727 -0.99735409 -0.07221254]]

    [[-0.38280325  0.06351545 -0.92164389]
    [-0.92379191 -0.03536405  0.3812583 ]
    [-0.00837727  0.99735409  0.07221254]]]
    ______________________ test_rops_frames[half_sphere_top] _______________________
    ----------------------------- Captured stdout call -----------------------------
    [[[ 0.21343543  0.82982397 -0.5155943 ]
    [ 0.01657105  0.52460431  0.85118489]
    [ 0.97681662 -0.19021695  0.0982181 ]]

    [[ 1.          0.         -0.        ]
    [ 0.          0.85065081  0.52573111]
    [ 0.         -0.52573111  0.85065081]]

    [[-0.5         0.80901699 -0.30901699]
    [-0.30901699 -0.5        -0.80901699]
    [-0.80901699 -0.30901699  0.5       ]]

    [[ 0.30901699  0.5         0.80901699]
    [-0.80901699 -0.30901699  0.5       ]
    [ 0.5        -0.80901699  0.30901699]]]
    """

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
    test_indices = np.array([0, 5, 10, 15])

    # Compare with individual LRF computations
    frames = framesh.rops.rops_frames_iterative(mesh, test_indices, radius=0.5)
    for frame, vertex_index in zip(frames, test_indices, strict=True):
        single_frame = framesh.rops.rops_lrf(mesh, vertex_index, radius=0.5)
        assert np.allclose(frame, single_frame)

    # Compare with individual LRF computations using vertex normals
    frames_with_normal = framesh.rops.rops_frames_iterative(
        mesh, test_indices, radius=2.0, use_vertex_normal=True
    )
    for frame, vertex_index in zip(frames_with_normal, test_indices, strict=True):
        single_frame = framesh.rops.rops_lrf(mesh, vertex_index, radius=2.0, use_vertex_normal=True)
        assert np.allclose(frame, single_frame)
