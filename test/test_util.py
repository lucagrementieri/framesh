import numpy as np
import trimesh.creation

from framesh.util import get_connected_nearby_indices


def test_get_connected_nearby_indices() -> None:
    cylinder = trimesh.creation.cylinder(radius=5.0, height=2.0)
    cylinder = cylinder.subdivide().subdivide().subdivide()
    _, vertex_index = cylinder.kdtree.query(np.array([0.0, 0.0, 1.0]))
    connected_indices = get_connected_nearby_indices(
        cylinder, vertex_index, radius=4.0, exclude_self=True
    )
    assert isinstance(connected_indices, np.ndarray)
    assert vertex_index not in connected_indices
    assert np.allclose(cylinder.vertices[connected_indices, 2], 1.0)
