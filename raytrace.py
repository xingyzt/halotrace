import numpy as np
from scipy.spatial import Voronoi, Delaunay, ConvexHull
from scipy.spatial.transform import Rotation
from time import time

def unwrap(coords):
    boxsize = 680000 # ckpc; TNG-Cluster periodic boundary size
    return (coords + (boxsize/2)) % boxsize - (boxsize/2)

def norm2(v): # norm squared
    return np.sum(v**2, axis=-1)

def normalize(v):
    assert np.any(v != 0)
    return v/np.sqrt(norm2(v))
    
def rot_to_z(v):
    """
    Returns the 3D rotation matrix R such that R.v / |v| = z = [0 0 1],

    v: ndarray of float, (3,)
    """

    # Ingest
    
    v = np.array(v, dtype=np.float64)
    assert v.shape == (3,)
    assert np.any(v != 0)

    # Rotate

    z = np.array([0, 0, 1]) # unit vector in z
    a = normalize(np.cross(v, z)) # rotation axis
    theta = np.arccos(normalize(np.dot(v, z))) # rotation angle
    rot = Rotation.from_rotvec(theta * a) # rotation object
    
    return rot.as_matrix() # rotation matrix


def impact(subhalo_ids, subhalo_centers, ray_pos, ray_dir, n_impacts):
    """
    Given a list of subhalo ids `subhalo_ids`, coordinates `subhalo_pos`,
    and a ray based at `ray_pos` in the direction `ray_dir`,
    returns a list of the `n_impacts`-closest (by impact parameter) 
    subhalo ids and impact parameters,
    sorted by order of encounter (earliest to latest).
    Indices are categorized by whether they lie in the forward or backwards rays.

    v: list of ndarray of float, shape (n_centers, 3)
    p: ndarray of float, shape (3,)
    n: ndarray of float, shape (3,)
    log: Boolean

    returns: tuple of ((front_ids, back_ids), (front_impacts, back_impacts))
    where
        *_ids: list of ndarray of int, shape (n_impacts,)
        *_impacts: list of ndarray of float, shape (n_impacts,3); 
    """

    # Ingest

    subhalo_ids = np.array(subhalo_ids)
    subhalo_centers = np.array(subhalo_centers, dtype=np.float64)
    n_centers = subhalo_centers.shape[0]
    ray_pos = np.array(ray_pos, dtype=np.float64)
    ray_dir = np.array(ray_dir, dtype=np.float64)
    z = np.array([0, 0, 1], dtype=np.float64) # unit vector in z
    assert subhalo_ids.shape == (n_centers,)
    assert subhalo_centers.shape == (n_centers, 3) 
    assert ray_pos.shape == ray_dir.shape == (3,)
    assert np.any(ray_dir != 0)

    zs = np.dot(subhalo_centers - ray_pos, ray_dir)  # projection onto ray axis

    # Calculate

    impacts = np.sqrt(norm2(
        (subhalo_centers - ray_pos) - zs[:, np.newaxis]*ray_dir
    ))

    front_select = zs > 0
    front_ids = subhalo_ids[front_select]
    front_impacts = impacts[front_select]
    back_ids = subhalo_ids[~front_select]
    back_impacts = impacts[~front_select]

    # Filter by impact parameter, then sort by time of impact
    front_sort = np.sort(np.argsort(front_impacts)[:n_impacts])
    back_sort = np.sort(np.argsort(back_impacts)[:n_impacts])

    return (
        (front_ids[front_sort], back_ids[back_sort]),
        (front_impacts[front_sort], back_impacts[back_sort])
    )

def sphere_intersect(v, r, p, n, log=True):
    """
    Given a list of spherical cell coordinates `v`, circumradii `r`,
    and a ray based at `p` in the direction `n`,
    returns the indices of all cells which intersects the ray,
    and their intersection lengths.
    Indices are categorized by whether they lie in the forward or backwards rays.

    v: list of ndarray of float, shape (n_centers, 3)
    r: list of ndarray of float, shape (n_centers,)
    p: ndarray of float, shape (3,)
    n: ndarray of float, shape (3,)
    log: Boolean

    returns: tuple of ((front_indices, back_indices), (front_lengths, back_lengths))
    where
        *_indices: list of ndarray of int, shape (n_intersecting_cells,)
        *_lengths: list of ndarray of float, shape (n_intersecting_cells,); 
    """

    # Ingest
    
    v = np.array(v, dtype=np.float64)
    r = np.array(r, dtype=np.float64)
    n_centers = v.shape[0]
    p = np.array(p, dtype=np.float64)
    n = np.array(n, dtype=np.float64)
    z = np.array([0, 0, 1], dtype=np.float64) # unit vector in z
    assert v.shape == (n_centers, 3) 
    assert p.shape == n.shape == (3,)
    assert np.any(n != 0)

    # Rotate then translate so that (p1, p2) gets sent to the z-axis
    if log: t0 = time()

    rot = rot_to_z(n) # rotation matrix
    trans = (rot @ p) # translation
    centers = (rot @ v.T).T - trans # transformed cell coordinates
    
    # Start with the cell closest to the origin, the find its neighbors by walking +z/-z
    if log: t1 = time()

    center_norm2 = norm2(centers)
    close_select = center_norm2 <= r**2
    close_sort = np.argsort(centers[close_select, 2])
    close_front_select = centers[close_select, 2][close_sort] > 0
    
    center_close_indices = np.arange(n_centers)[close_select][close_sort]
    lengths = 2*np.sqrt(r[close_select]**2 - center_norm2[close_select])[close_sort]
    
    if log: t2 = time()
    if log:
        print('Total:', t2 - t0, 's')
        print('Transform:', t1 - t0, 's')
        print('Truncate:', t2 - t1, 's')

    return (
        (
            center_close_indices[ close_front_select],
            center_close_indices[~close_front_select],
        ), (
            lengths[ close_front_select],
            lengths[~close_front_select],
        )
    )


def voronoi_intersect(v, r, p, n, log=True):
    """
    Given a list of spherical cell coordinates `v`, circumradii `r`,
    and a ray based at `p` in the direction `n`,
    returns the indices of all cells which intersects the ray,
    and their intersection lengths.
    Indices are categorized by whether they lie in the forward or backwards rays.

    v: list of ndarray of float, shape (n_centers, 3)
    r: list of ndarray of float, shape (n_centers,)
    p: ndarray of float, shape (3,)
    n: ndarray of float, shape (3,)
    log: Boolean

    returns: tuple of ((front_indices, back_indices), (front_lengths, back_lengths))
    where
        *_indices: list of ndarray of int, shape (n_intersecting_cells,)
        *_lengths: list of ndarray of float, shape (n_intersecting_cells,); 
    """

    # Ingest
    
    v = np.array(v, dtype=np.float64)
    r = np.array(r, dtype=np.float64)
    n_centers = v.shape[0]
    p = np.array(p, dtype=np.float64)
    n = np.array(n, dtype=np.float64)
    z = np.array([0, 0, 1], dtype=np.float64) # unit vector in z
    assert v.shape == (n_centers, 3) 
    assert p.shape == n.shape == (3,)
    assert np.any(n != 0)

    # Rotate then translate so that (p1, p2) gets sent to the z-axis
    if log: t0 = time()

    rot = rot_to_z(n) # rotation matrix
    trans = (rot @ p) # translation
    centers = (rot @ v.T).T - trans # transformed Voronoi cell coordinates
    
    # Start with the cell closest to the origin, then find its neighbors by walking +z/-z
    if log: t1 = time()

    close_select = norm2(centers[:, :2]) < r*r
    center_close_indices = np.arange(n_centers)[close_select]
    close = centers[close_select]
    n_close = close.shape[0]

    if n_close == 0:
        return ([], [])

    if log: t2 = time()

    vor = Voronoi(close)
    n_ridges = vor.ridge_points.shape[0]

    if log: t3 = time()

    # initial index in close
    j0 = np.argmin(norm2(close))
    
    pair_indices = np.arange(n_ridges*2, dtype=np.int64)
    close_pair_indices = vor.ridge_points.T.flatten()

    bi_indices = ([], [])
    bi_zs = ([], [])
    bi_signs = (-1, 1)

    bi_center_indices = []
    bi_lengths = []

    
    for (i, sign) in enumerate(bi_signs):
        
        j = j0
        z_crossing = 0
        
        while j != -1:

            bi_indices[i].append(j)
            bi_zs[i].append(z_crossing)

            pair_self_indices = pair_indices[close_pair_indices == j]
            pair_neighbor_indices = pair_self_indices - n_ridges
            
            close_neighbor_indices = close_pair_indices[pair_neighbor_indices]
            forward_select = sign*close[close_neighbor_indices,2] > sign*close[j,2] # cells above/below
            close_forward_indices = close_neighbor_indices[forward_select]

            ridge_forward_indices = (pair_self_indices % n_ridges)[forward_select]
            
            j = -1
            for (forward_index, ridge_forward_index) in enumerate(ridge_forward_indices):
        
                ridge_verts = vor.vertices[vor.ridge_vertices[ridge_forward_index]]
                ridge_2d = Delaunay(ridge_verts[:, :2]) # tris
                tri = ridge_2d.find_simplex((0,0))
                if tri == -1: continue
        
                tri_verts = ridge_verts[ridge_2d.simplices[tri]]
                coeffs = np.linalg.inv(tri_verts.T) @ np.array([0, 0, 1], dtype=np.float64)
                z_crossing = 1/np.sum(coeffs)
                
                if sign*z_crossing < sign*bi_zs[i][-1]: break # wrong direction

                j = close_forward_indices[forward_index]
                break

        bi_center_indices.append(center_close_indices[bi_indices[i]][:-1])
        bi_lengths.append(sign * np.diff(bi_zs[i]))

    if log: t4 = time()
    if log:
        print('Total:', t4 - t0, 's')
        print('Transform:', t1 - t0, 's')
        print('Truncate:', t2 - t1, 's')
        print('Voronoi:', t3 - t2, 's')
        print('Walk:', t4 - t3, 's')

    return (bi_center_indices, bi_lengths)