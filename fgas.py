import numpy as np
import h5py
import time

import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as const
import matplotlib.pyplot as plt
import raytrace
import sys

rng = np.random.default_rng(seed=1)

sub_index = int(sys.argv[1])
repeat = True

with h5py.File("outputs/dm_proto.hdf5", "r") as f:
    halo_ids = list(f.keys())

halo_id = halo_ids[sub_index][5:]
print(halo_id)

with h5py.File("outputs/dm_proto.hdf5", "r") as f:
    halo = f['halo_'+halo_id]
    com = halo['GroupCM'][:]
    halo_pos = halo['GroupPos'][:]
    if not repeat and 'Group_Fgas_Crit200' in halo.keys():
        print('halo already calculated')
        sys.exit(0)

    r200 = halo['Group_R_Crit200'][()]
    r500 = halo['Group_R_Crit500'][()]
    
    m200 = halo['Group_M_Crit200'][()]
    m500 = halo['Group_M_Crit500'][()]
    mgas = halo['GroupMassType'][0]

    # subs = halo['subhalos']
    # sub0_cm = np.array([ subs['cm_x'][0], subs['cm_y'][0], subs['cm_z'][0] ]).T

with h5py.File(f'tng_cache/snap_099/cutout_{halo_id}.hdf5', 'r') as f:
    gas_pos = raytrace.unwrap(f['PartType0/Coordinates'][:] - halo_pos) # ckpc
    gas_mass = f['PartType0/Masses'][:] # Msun
    
    eta_e = f['PartType0/ElectronAbundance'][:] # fraction
    X_H = f['PartType0/GFM_Metals'][:,0] # fraction
    gas_m_e = gas_mass * eta_e * X_H # Msun

r = np.linalg.norm(gas_pos, axis=-1)
fgas200 = np.sum(gas_mass[r < r200])/m200
fgas500 = np.sum(gas_mass[r < r500])/m500
fe200 = np.sum(gas_m_e[r < r200])/m200
fe500 = np.sum(gas_m_e[r < r500])/m500

print(np.sum(r>-1))
print(np.sum(r<r500))
print(fgas500)
# plt.hist(r)

sys.exit(0)

with h5py.File("outputs/dm_proto_sfr_weighted.hdf5", "a") as f:
    halo = f['halo_'+halo_id]
    halo['Group_Fgas_Crit500_max'] = mgas/m500
    if 'Group_Fgas_Crit200' in halo.keys():
        del halo['Group_Fgas_Crit200']
        del halo['Group_Fgas_Crit500']
        del halo['Group_Fe_Crit200']
        del halo['Group_Fe_Crit500']
    halo['Group_Fgas_Crit200'] = fgas200
    halo['Group_Fgas_Crit500'] = fgas500
    halo['Group_Fe_Crit200'] = fe200
    halo['Group_Fe_Crit500'] = fe500
