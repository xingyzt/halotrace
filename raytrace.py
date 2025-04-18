import numpy as np
from scipy.spatial import Voronoi, Delaunay, ConvexHull
from scipy.spatial.transform import Rotation

def rot_to_z(v):
    """
    Returns the 3D rotation matrix R such that R.v / |v| = z = [0 0 1],

    v: ndarray of float, (3,)
    """

    # Ingest
    
    v = np.array(v, dtype=np.float64)
    zeros = np.zeros(3, dtype=np.float64)
    assert v.shape == (3,)
    assert np.any(v != zeros)

    # Rotate

    z = np.array([0, 0, 1]) # unit vector in z
    a = np.cross(v, z) # rotation axis
    if np.any(a != zeros):
        a /= np.sqrt(np.sum(a*a))  # normalized rotation axis
    theta = np.arccos(np.dot(v, z) / np.sqrt(np.sum(v*v))) # rotation angle
    rot = Rotation.from_rotvec(theta * a) # rotation object
    
    return rot.as_matrix() # rotation matrix

def intersect(v, p1, p2):
    """
    Given a list of Voronoi cell coordinates `v`,
    and a ray which intersects the points `p1` and `p2`,
    returns the indices of the cells which intersects the ray.

    v: list of ndarray of float, (n_cells, 3)
    p1: ndarray of float, (3,)
    p2: ndarray of float, (3,)

    returns: list of ndarray of int, (n_intersecting_cells,)
    """

    # Ingest
    
    v = np.array(v, dtype=np.float64)
    n_cells = v.shape[0]
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    z = np.array([0, 0, 1], dtype=np.float64) # unit vector in z
    assert v.shape == (n_cells, 3) 
    assert p1.shape == p2.shape == (3,)
    assert np.any(p1 != p2)

    # Rotate then translate so that (p1, p2) gets sent to the z-axis

    rot = rot_to_z(p2 - p1) # rotation matrix
    trans = (rot @ p1) * (1 - z) # translation
    centers = (rot @ v.T).T - trans # transformed Voronoi cell coordinates

    # Find the cells which intersect the z-axis,
    # by projecting each cell onto the xy-plane,
    # then checking if they contain (0,0).
    
    vor = Voronoi(centers)
    indices = []
    for (i, j) in enumerate(vor.point_region): # (index of center, index of cell around center)
        cell = vor.regions[j]
        if -1 in cell: continue # ignores voronoi ridges (infinite cells). TODO: include them
        verts = vor.vertices[cell][:,:2] # vertices of cells, with z-component removed
        if not verts.size: continue # some cells do not have vertices
        hull = ConvexHull(verts)
        tris = Delaunay(verts[hull.vertices])
        intersect = tris.find_simplex((0,0)) != -1
        if intersect:
            indices.append(i)

    return np.array(indices, dtype=np.int64)