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

def voronoi_cell_on_z(vor, i):
    """
    Check if the i-th Voronoi center has a cell that intersects the z-axis,
    by moving its cell's barycenter to the xy-plane and checking if the cell contains the origin.
    Returns the crossing length.

    vor: scipy.spatial.Voronoi
    i: int

    returns: float
    """

    cell = vor.regions[vor.point_region[i]]
    if not cell or -1 in cell:  # ignores voronoi ridges (infinite cells). TODO: include them
        return -1
        
    verts = vor.vertices[cell]  # vertices of cells
    z_mean = np.mean(verts[:,2], axis=0)
    
    tetras = Delaunay(verts) # triangulation with tetrahedrons
    if tetras.find_simplex((0,0,z_mean)) == -1:
        return -1  # no crossing
    else:
        faces = tetras.convex_hull
        interator = (
            face for face in faces
            if Delaunay(verts[face][:, :2]).find_simplex((0,0)) != -1
        )
        faces_on_z = ( next(interator), next(interator) ) # only two faces that cross
        
        zs = []
        for face in faces_on_z:
            tri_verts = verts[face] # vertices of triangle
            coeffs = np.linalg.inv(tri_verts.T) @ np.array([0, 0, 1], dtype=np.float64)
            zs.append(1/np.sum(coeffs))
        
        return max(zs) - min(zs)

def sphere_intersect(v, r, p1, p2, log=True):
    """
    Given a list of spherical cell coordinates `v`, circumradii `r`,
    and a ray which intersects the points `p1` and `p2`,
    returns the indices of all cells which intersects the ray,
    and their intersection lengths.

    v: list of ndarray of float, shape (n_centers, 3)
    r: list of ndarray of float, shape (n_centers,)
    p1: ndarray of float, shape (3,)
    p2: ndarray of float, shape (3,)
    log: Boolean

    returns: tuple of (indices, lengths)
    indices: list of ndarray of int, shape (n_intersecting_cells,)
    lengths: list of ndarray of float, shape (n_intersecting_cells,); 
    """

    # Ingest
    
    v = np.array(v, dtype=np.float64)
    r = np.array(r, dtype=np.float64)
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

    center_norm2 = norm2(centers[:, :2])
    close_select = center_norm2 <= r**2
    center_close_indices = np.arange(n_centers)[close_select]
    lengths = 2*np.sqrt(r[close_select]**2 - center_norm2[close_select])

    t2 = time()
    if log:
        print('Total:', t2 - t0, 's')
        print('Transform:', t1 - t0, 's')
        print('Truncate:', t2 - t1, 's')

    return (center_close_indices, lengths)


def voronoi_intersect(v, r, p1, p2, log=True):
    """
    Given a list of spherical cell coordinates `v`, circumradii `r`,
    and a ray which intersects the points `p1` and `p2`,
    returns the indices of all cells which intersects the ray,
    and their intersection lengths.

    v: list of ndarray of float, shape (n_centers, 3)
    r: list of ndarray of float, shape (n_centers,)
    p1: ndarray of float, shape (3,)
    p2: ndarray of float, shape (3,)
    log: Boolean

    returns: tuple of (indices, lengths)
    indices: list of ndarray of int, shape (n_intersecting_cells,)
    lengths: list of ndarray of float, shape (n_intersecting_cells,); 
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

    close_select = norm2(centers[:, :2]) < (r**2)
    center_close_indices = np.arange(n_centers)[close_select]
    close = centers[close_select]
    n_close = close.shape[0]

    t2 = time()

    vor = Voronoi(close)

    t3 = time()

    # initial index in close
    j0 = np.argmin(norm2(close[:, :2]))

    # set of indices and lengths encountered walking +z, -z
    bi_indices = ([], [])
    bi_lengths = ([], [])
    bi_signs = (-1, 1)
    
    for (i, sign) in enumerate(bi_signs):
        j = j0
        length = 0
        while j != -1:
            if j != j0:
                bi_indices[i].append(j)
                bi_lengths[i].append(length)
            forward_select = sign*close[:,2] > sign*close[j,2] # cells above/below
            forward_indices_in_close = np.arange(n_close)[forward_select]
            dists = norm2(close[forward_select] - close[j])
            trial_forward_indices = np.argsort(dists) # TODO: use lazysort
            j, length = next(( 
                (k, voronoi_cell_on_z(vor, k)) for k in forward_indices_in_close[trial_forward_indices]
                if voronoi_cell_on_z(vor, k) > 0
            ), (-1, 0))
    
    close_indices_on_z = np.array([
        *reversed(bi_indices[0]), 
        j0,
        *bi_indices[1]
    ], dtype=np.int64)
    
    lengths = np.array([
        *reversed(bi_lengths[0]), 
        voronoi_cell_on_z(vor, j0), 
        *bi_lengths[1]
    ], dtype=np.float64)

    t4 = time()
    if log:
        print('Total:', t4 - t0, 's')
        print('Transform:', t1 - t0, 's')
        print('Truncate:', t2 - t1, 's')
        print('Voronoi:', t3 - t2, 's')
        print('Walk:', t4 - t3, 's')

    center_indices_on_z = center_close_indices[close_indices_on_z]
    return (center_indices_on_z, lengths)