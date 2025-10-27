#!/usr/bin/env python3
import numpy as np
import h5py
import time

import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as const

import raytrace
import sys

import tqdm

rng = np.random.default_rng(seed=1)

halo_index = int(sys.argv[1])
print(halo_index)

repeat = True

path = 'outputs/tng_cluster_catalog.hdf5'
# file = 'outputs/dm_proto_sfr_weighted.hdf5'

kpc_per_cm = 3.2407792894443653e-22

with h5py.File(path, "r") as f:
    halo_id = f['table/haloID'][halo_index]

    if not repeat and f'trace/halo_{halo_id}' in f:
        print('halo already traced')
        sys.exit(0)

    com = f['table/GroupCM'][halo_index]
    center = f['table/GroupPos'][halo_index]
    r500 = f['table/Group_R_Crit500'][halo_index]
    # subs = halo['subhalos']
    # sub_ids = subs['id'][:]
    # sub_coms = raytrace.unwrap(np.array([ subs['cm_x'], subs['cm_y'], subs['cm_z'] ]).T - com)

with h5py.File(f'tng_cache/snap_099/cutout_{halo_id}.hdf5', 'r') as f:
    m_g = f['PartType0/Masses'][:] * (1e10 * u.solMass).to(u.kg) # kg
    rho_g = f['PartType0/Density'][:] * (1e10 * u.solMass).to(u.kg) / cu.littleh / ( u.kpc / cu.littleh)**3 # kg ckpc^-3
    V = m_g/rho_g
    r = ((0.75/np.pi)*V)**(1/3)
    eta_e = f['PartType0/ElectronAbundance'][:] # fraction
    X_H = f['PartType0/GFM_Metals'][:,0] # fraction
    # X_neutral_H = f['PartType0/NeutralHydrogenAbundance'][:]
    
    star_select = f['PartType4/GFM_StellarFormationTime'][:] > 0 # select only non-wind stuff
    star_pos = raytrace.unwrap(f['PartType4/Coordinates'][star_select] - com) # ckpc
    star_id = f['PartType4/ParticleIDs'][star_select] # id
    # star_subhalo_id = f['PartType4/SubhaloIDs'][star_select] # id
    star_mass = f['PartType4/Masses'][star_select] * 1e10 # Msun
    # star_scalefactor = f['PartType4/GFM_StellarFormationTime'][star_select]
    # print(np.min(star_scalefactor))
    # star_formationtime = np.log(star_scalefactor) * 14e9 # years, energy-dominated era
    # star_sfr = star_mass / star_formationtime
    
    gas_pos = raytrace.unwrap(f['PartType0/Coordinates'][:] - com) # ckpc
    gas_id = f['PartType0/ParticleIDs'][:] # id
    # gas_subhalo_id = f['PartType0/SubhaloIDs'][:] # id
    gas_mass = f['PartType0/Masses'][:] * 1e10 # Msun
    gas_sfr = f['PartType0/StarFormationRate'][:] # Msun/yr
    gas_n_e = (rho_g * eta_e * X_H / const.m_p).value # ckpc^-3


if np.sum(gas_sfr) == 0:
    sys.exit()

n_samples = 2 * 200
# n_flybys = 4 # number of subhalo n_flybys to count, including origin subhalo
n_markers = 5 # number of distances to measure DM
markers = np.append(10 ** np.arange(n_markers), np.inf) # from 10^0 ckpc to 10^4 ckpc
voronoi = True
dm_method = raytrace.voronoi_intersect if voronoi else raytrace.sphere_intersect
dm_name = 'dm' if voronoi else 'dm_sphere'
data = {
    'index': np.arange(n_samples, dtype=np.int64),
    'host_id': np.zeros(n_samples, dtype=np.int64),
    'host_pos': np.zeros((n_samples, 3), dtype=np.float64),
    # 'host_ssfr': np.zeros(n_samples, dtype=np.float64),
    'direction': np.zeros((n_samples, 3), dtype=np.float64),
    # 'subhalo_id': np.zeros((n_samples, n_flybys), dtype=np.int64),
    # 'subhalo_impact': np.zeros((n_samples, n_flybys, 3), dtype=np.float64),
    dm_name: np.zeros((n_samples, n_markers + 1), dtype=np.float64),
}
# z_history = []
# n_e_history = []

log_interval = 100

for weight in ('sfr', 'mstar'):
    print(weight)
    for i in tqdm.trange(n_samples//2):
        
        if weight == 'sfr':
            
            host_index = rng.choice(gas_id.shape[0], p=gas_sfr/np.sum(gas_sfr))
        
            ray_host_id = gas_id[host_index]
            ray_host_pos = gas_pos[host_index]
            ray_host_mass = gas_mass[host_index]
            
        elif weight == 'mstar':
            
            host_index = rng.choice(star_id.shape[0], p=star_mass/np.sum(star_mass))
        
            ray_host_id = star_id[host_index]
            ray_host_pos = star_pos[host_index]
            ray_host_mass = star_mass[host_index]
    
        ray_dir = raytrace.normalize(rng.normal(size=3))
    
        bi_intersects, bi_lengths = dm_method(gas_pos, 5*r.value, ray_host_pos, ray_dir, log=False)
    
        # get impact parameters to host subhalo, and 3 most massive subhalos in cluster
        # impact_sub_ids = np.array([ ray_subhalo_id, sub_ids[0], sub_ids[1], sub_ids[2] ])
        # impact_sub_coms = np.array([ 
        #     ( sub_coms[sub_ids == impact_sub_id][0]  if impact_sub_id != -1 else np.zeros((3,)) )
        #     for impact_sub_id in impact_sub_ids
        # ]) 
        # impacts = raytrace.impact(
        #     impact_sub_coms, ray_host_pos, ray_dir
        # )
    
        for j in (0, 1): 
            ij = 2*i + j
            
            data['host_id'  ][ij] = ray_host_id
            data['host_pos' ][ij] = ray_host_pos
            data['direction'][ij] = (2*j-1) * ray_dir
    
            intersects = bi_intersects[j]
            lengths = bi_lengths[j]
            distance = np.cumsum(lengths)
            marker_selects = distance < markers[:, np.newaxis]
    
            # print(lengths.shape, intersects.shape)
            # plt.plot(lengths)
            # plt.show()
        
            trace_dm = lengths * gas_n_e[intersects] * 1000 * (kpc_per_cm ** 3) # pc/cm3
            # trace_sub_ids = gas_subhalo_id[intersects]
    
            # for (k, impact_sub_id) in enumerate(impact_sub_ids):
            #     data['subhalo_id'    ][ij, k] = impact_sub_id
            #     data['subhalo_impact'][ij, k] = impacts[k]
            #     data[dm_name         ][ij, k] = np.array([
            #         np.sum(trace_dm[marker_select & (trace_sub_ids==impact_sub_id)])
            #         for marker_select in marker_selects
            #     ])
                
            data[dm_name][ij] = np.array([
                np.sum(trace_dm[marker_select])
                for marker_select in marker_selects
            ]) # total DM, including halo fuzz
    
            # z_history.append(np.cumsum(lengths))
            # n_e_history.append(gas_n_e[intersects])
        
        # if not i % log_interval:
        #     if i > 0:
        #         print(i, (time.time() - t)/log_interval)
        #     t = time.time()

    with h5py.File(path, "a") as f:
        out = f'trace_{weight}/halo_{halo_id}'
        if out in f:
            del f[out]
        f.create_group(out)
        for k in data:
            f[out][k] = data[k]
