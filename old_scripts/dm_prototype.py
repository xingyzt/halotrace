import numpy as np
import matplotlib.pyplot as plt
import h5py
import time

import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as const

import raytrace

n_samples = 2 * 1000
n_flybys = 4 # number of subhalo n_flybys to count, including origin subhalo
n_markers = 5 # number of distances to measure DM
log_interval = 100

markers = np.append(10 ** np.arange(n_markers), np.inf) # from 10^0 ckpc to 10^4 ckpc
voronoi = True
dm_method = raytrace.voronoi_intersect if voronoi else raytrace.sphere_intersect
dm_name = 'dm_voronoi' if voronoi else 'dm_sphere'

rng = np.random.default_rng(seed=1)

with h5py.File("outputs/dm_proto.hdf5", "r") as f:
    halo_ids = [key[5:] for key in f.keys()]

for halo_id in halo_ids[5:]:

    print(halo_id)
    
    with h5py.File("outputs/dm_proto.hdf5", "r") as f:
        halo = f['halo_'+halo_id]
        com = halo['GroupCM'][:]
        r500 = halo['Group_R_Crit500'][()]
        subs = halo['subhalos']
        sub_ids = subs['id'][:]
        sub_coms = raytrace.unwrap(np.array([ subs['cm_x'], subs['cm_y'], subs['cm_z'] ]).T - com)

    with h5py.File(f'tng_cache/snap_099/cutout_{halo_id}.hdf5', 'r') as f:

        m_g = f['PartType0/Masses'][:] * (1e10 * u.solMass).to(u.kg) # kg
        rho_g = f['PartType0/Density'][:] * (1e10 * u.solMass).to(u.kg) / cu.littleh / ( u.kpc / cu.littleh)**3 # kg ckpc^-3
        V = m_g/rho_g
        r = ((0.75/np.pi)*V)**(1/3)
        eta_e = f['PartType0/ElectronAbundance'][:] # fraction
        X_H = f['PartType0/GFM_Metals'][:,0] # fraction
        
        gas_pos = raytrace.unwrap(f['PartType0/Coordinates'][:] - com) # ckpc
        gas_id = f['PartType0/ParticleIDs'][:] # id
        gas_subhalo_id = f['PartType0/SubhaloIDs'][:] # id
        gas_mass = f['PartType0/Masses'][:] * 1e10 # Msun
        gas_sfr = f['PartType0/StarFormationRate'][:] # Msun/yr
        gas_n_e = (rho_g * eta_e * X_H / const.m_p).value # ckpc^-3

    data = {
        'indices': np.arange(n_samples, dtype=np.int64),
        'star_id': np.zeros(n_samples, dtype=np.int64),
        'star_pos': np.zeros((n_samples, 3), dtype=np.float64),
        'star_ssfr': np.zeros(n_samples, dtype=np.float64),
        'direction': np.zeros((n_samples, 3), dtype=np.float64),
        'subhalo_id': np.zeros((n_samples, n_flybys), dtype=np.int64),
        'subhalo_impact': np.zeros((n_samples, n_flybys, 3), dtype=np.float64),
        dm_name: np.zeros((n_samples, n_flybys + 1, n_markers + 1), dtype=np.float64),
    }
    z_history = []
    n_e_history = []

    pdf = gas_sfr/np.sum(gas_sfr)
    for i in range(n_samples//2):
        star_index = rng.choice(gas_id.shape[0], p=pdf)
    
        ray_star_id = gas_id[star_index]
        ray_star_pos = gas_pos[star_index]
        ray_star_sfr = gas_sfr[star_index]
        ray_star_mass = gas_mass[star_index]
        ray_dir = raytrace.normalize(rng.normal(size=3))
    
        bi_intersects, bi_lengths = dm_method(gas_pos, 5*r.value, ray_star_pos, ray_dir, log=False)
        bi_ids, bi_impacts = raytrace.impact(
            sub_ids, sub_coms, ray_star_pos, ray_dir, n_flybys
        )
    
        for j in (0, 1): 
            ij = 2*i + j
            
            data['star_id'  ][ij] = ray_star_id
            data['star_pos' ][ij] = ray_star_pos
            data['star_ssfr'][ij] = ray_star_sfr/ray_star_mass
            data['direction'][ij] = (ray_dir) if (j == 0) else (-ray_dir)
    
            intersects = bi_intersects[j]
            lengths = bi_lengths[j]
            distance = np.cumsum(lengths)
            marker_selects = distance < markers[:, np.newaxis]
    
            # print(lengths.shape, intersects.shape)
            # plt.plot(lengths)
            # plt.show()
        
            trace_dm = lengths * gas_n_e[intersects]
            trace_ids = gas_subhalo_id[intersects]
    
            for (k, close_id) in enumerate(bi_ids[j]):
                data['subhalo_id'    ][ij, k] = close_id
                data['subhalo_impact'][ij, k] = bi_impacts[j][k]
                data[dm_name         ][ij, k] = np.array([
                    np.sum(trace_dm[marker_select & (trace_ids==close_id)])
                    for marker_select in marker_selects
                ])
                
            data[dm_name][ij, -1] = np.array([
                np.sum(trace_dm[marker_select])
                for marker_select in marker_selects
            ]) # total DM, including halo fuzz
    
            z_history.append(np.cumsum(lengths))
            n_e_history.append(gas_n_e[intersects])
        
        if not i % log_interval:
            if i > 0:
                print(halo_id, i, (time.time() - t)/log_interval)
            t = time.time()

    with h5py.File("outputs/dm_proto.hdf5", "a") as f:
        
        halo = f['halo_' + halo_id]
        if 'trace' in halo.keys():
            del halo['trace']
        halo.create_group('trace')
        for k in data:
            halo['trace'][k] = data[k]
