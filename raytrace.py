import numpy as np
from scipy.spatial import Voronoi, Delaunay, ConvexHull
from scipy.spatial.transform import Rotation
from time import time

def norm2(v): # norm squared
    return np.sum(v**2, axis=-1)
    
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
        a /= np.sqrt(norm2(a))  # normalized rotation axis
    theta = np.arccos(np.dot(v, z) / np.sqrt(norm2(v))) # rotation angle
    rot = Rotation.from_rotvec(theta * a) # rotation object
    
    return rot.as_matrix() # rotation matrix

def cell_on_z(vor, i):
    """
    Check if the i-th Voronoi center has a cell that intersects the z-axis,
    by projecting it cell onto the xy-plane then checking if it contains (0,0).

    vor: scipy.spatial.Voronoi
    i: int

    returns: Boolean
    """
    cell = vor.regions[vor.point_region[i]]
    if not cell or -1 in cell:  # ignores voronoi ridges (infinite cells). TODO: include them
        return False
    verts = vor.vertices[cell][:, :2]  # vertices of cells, with z-component removed
    hull = ConvexHull(verts)
    tris = Delaunay(verts[hull.vertices])
    return tris.find_simplex((0, 0)) != -1


def intersect(v, p1, p2, max_dist=100, log=True):
    """
    Given a list of Voronoi cell coordinates `v`,
    and a ray which intersects the points `p1` and `p2`,
    returns the indices of all cells which intersects the ray,
    which are within `max_dist` from the ray.

    v: list of ndarray of float, (n_centers, 3)
    p1: ndarray of float, (3,)
    p2: ndarray of float, (3,)
    max_dist: float
    log: Boolean

    returns: list of ndarray of int, (n_intersecting_cells,)
    """

    # Ingest
    
    v = np.array(v, dtype=np.float64)
    n_centers = v.shape[0]
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    z = np.array([0, 0, 1], dtype=np.float64) # unit vector in z
    assert v.shape == (n_centers, 3) 
    assert p1.shape == p2.shape == (3,)
    assert np.any(p1 != p2)

    # Rotate then translate so that (p1, p2) gets sent to the z-axis
    t0 = time()

    rot = rot_to_z(p2 - p1) # rotation matrix
    trans = (rot @ p1) * (1 - z) # translation
    centers = (rot @ v.T).T - trans # transformed Voronoi cell coordinates
    
    # Start with the cell closest to the z-axis, the find its neighbors by walking +z/-z
    t1 = time()

    close_select = norm2(centers[:, :2]) < (max_dist**2)
    center_close_indices = np.arange(n_centers)[close_select]
    close = centers[close_select]
    n_close = close.shape[0]

    t2 = time()

    vor = Voronoi(close)

    t3 = time()
    
    close_indices_on_z = [ np.argmin(norm2(close[:, :2])) ]
    for sign in (-1, 1):
        i = close_indices_on_z[0]
        while i != -1:
            close_indices_on_z.append(i)
            forward_select = sign*close[:,2] > sign*close[i,2] # cells above/below
            forward_indices_in_close = np.arange(n_close)[forward_select]
            dists = norm2(close[forward_select] - close[i])
            trial_forward_indices = np.argsort(dists) # TODO: use lazysort
            i = next(( 
                k for k in forward_indices_in_close[trial_forward_indices ]
                if cell_on_z(vor, k) 
            ), -1)

    t4 = time()
    if log:
        print('Total:', t4 - t0, 's')
        print('Transform:', t4 - t3, 's')
        print('Truncate:', t3 - t2, 's')
        print('Voronoi:', t2 - t1, 's')
        print('Walk:', t3 - t2, 's')

    center_indices_on_z = center_close_indices[close_indices_on_z]
    return center_indices_on_z