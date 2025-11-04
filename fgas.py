import numpy as np
import h5py
import time

import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as const
import matplotlib.pyplot as plt
import raytrace
import sys

import tqdm

rng = np.random.default_rng(seed=1)

halo_index = int(sys.argv[1])
repeat = True

with h5py.File("outputs/tng_cluster_catalog.hdf5", "r") as f:
    halo_id = f['table/haloID'][halo_index]
    print(halo_id)

    halo_pos = f['table/GroupPos'][halo_index]
    if not repeat and f['table/Group_Fgas_Crit200'][halo_index] > 1e-6:
        print('halo already calculated')
        sys.exit(0)

    r200 = f['table/Group_R_Crit200'][halo_index]
    r500 = f['table/Group_R_Crit500'][halo_index]
    
    m200 = f['table/Group_M_Crit200'][halo_index]
    m500 = f['table/Group_M_Crit500'][halo_index]
    mgas = f['table/GroupMassType'][halo_index]

    # subs = halo['subhalos']
    # sub0_cm = np.array([ subs['cm_x'][0], subs['cm_y'][0], subs['cm_z'][0] ]).T

with h5py.File(f'tng_cache/snap_099/cutout_{halo_id}.hdf5', 'r') as f:
    gas_pos = raytrace.unwrap(f['PartType0/Coordinates'][:] - halo_pos) # ckpc
    gas_mass = f['PartType0/Masses'][:] # Msun
    
    eta_e = f['PartType0/ElectronAbundance'][:] # fraction
    X_H = f['PartType0/GFM_Metals'][:,0] # fraction
    gas_m_ion = gas_mass * eta_e * X_H # Msun

r = np.linalg.norm(gas_pos, axis=-1)
fgas200 = np.sum(gas_mass[r < r200])/m200
fgas500 = np.sum(gas_mass[r < r500])/m500
fe200 = np.sum(gas_m_ion[r < r200])/m200
fe500 = np.sum(gas_m_ion[r < r500])/m500

# Sort by radius once
idx = np.argsort(r)
r_sorted = r[idx]
gas_mass_sorted = gas_mass[idx]
gas_m_ion_sorted = gas_m_ion[idx]

cum_mgas = np.cumsum(gas_mass_sorted)
cum_mion = np.cumsum(gas_m_ion_sorted)

# Radii to evaluate
Rs = 10 ** np.linspace(0, 5, 1001)[1:]

# Find insertion indices for Rs in sorted radii
inds = np.searchsorted(r_sorted, Rs, side="right")

mgas = cum_mgas[np.clip(inds - 1, 0, len(cum_mgas) - 1)]
mion = cum_mion[np.clip(inds - 1, 0, len(cum_mion) - 1)]

# print(np.sum(r>-1))
# print(np.sum(r<r500))
# print(fgas500)
# print(Rs, mgas)
# plt.hist(r)
# sys.exit()

with h5py.File("outputs/tng_cluster_catalog.hdf5", "a") as f:
    size = f['table/haloID'].size
    if 'table/Group_R_enclosed' in f:
        del f['table/Group_R_enclosed']
    if 'table/Group_Mgas_enclosed' in f:
        del f['table/Group_Mgas_enclosed']
    if 'table/Group_Mion_enclosed' in f:
        del f['table/Group_Mion_enclosed']
    f.require_dataset('table/Group_R_enclosed', shape=(size, len(Rs)), dtype='f8')
    f.require_dataset('table/Group_Mgas_enclosed', shape=(size, len(Rs)), dtype='f8')
    f.require_dataset('table/Group_Mion_enclosed', shape=(size, len(Rs)), dtype='f8')
    f['table/Group_R_enclosed'][halo_index] = Rs
    f['table/Group_Mgas_enclosed'][halo_index] = mgas
    f['table/Group_Mion_enclosed'][halo_index] = mion
    f['table/Group_Fgas_Crit200'][halo_index] = fgas200
    f['table/Group_Fgas_Crit500'][halo_index] = fgas500
    f['table/Group_Fe_Crit200'][halo_index] = fe200
    f['table/Group_Fe_Crit500'][halo_index] = fe500
