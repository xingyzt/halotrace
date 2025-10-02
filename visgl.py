import numpy as np
import vispy.scene
from vispy.scene import visuals

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, LogNorm
import h5py
import time
import os
import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as const
import raytrace
import sys

halo_index = int(sys.argv[1])

with h5py.File("outputs/dm_proto.hdf5", "r") as f:
    halo_ids = list(f.keys())

    halo_id = halo_ids[halo_index][5:]
    print(halo_id)

    halo = f['halo_' + halo_id]

    print(halo['GroupMassType'][:])
    com = halo['GroupCM'][:]
    r500 = halo['Group_R_Crit500'][()]
    subhalo_id = halo['subhalos']["id"][:]

with h5py.File(f'tng_cache/snap_099/cutout_{halo_id}.hdf5', 'r') as f:
    gas_mass = f['PartType0/Masses'][:] * ((1e10 * u.solMass).to(u.kg) / cu.littleh) # kg ckpc^-3
    gas_density = f['PartType0/Density'][:] * (1e10 * u.solMass).to(u.kg) / cu.littleh / ( u.kpc / cu.littleh)**3 # kg ckpc^-3
    gas_volume = gas_mass/gas_density
    gas_radius = ((0.75/np.pi)*gas_volume)**(1/3)
    gas_electron_abundance = f['PartType0/ElectronAbundance'][:] # fraction
    gas_hydrogen_massfrac = f['PartType0/GFM_Metals'][:,0] # fraction
    gas_sfr = f['PartType0/StarFormationRate'][:] # Msun/year
    
    gas_pos = raytrace.unwrap(f['PartType0/Coordinates'][:] - com) # ckpc
    gas_id = f['PartType0/ParticleIDs'][:] # id
    gas_subhalo_id = f['PartType0/SubhaloIDs'][:] # id
    gas_sfr = f['PartType0/StarFormationRate'][:] # Msun/yr
    gas_n_e = (gas_density * gas_electron_abundance * gas_hydrogen_massfrac / const.m_p).value # ckpc^-3
    gas_N_e = (gas_mass * gas_electron_abundance * gas_hydrogen_massfrac / const.m_p).value # ckpc^-3
    
    star_select = f['PartType4/GFM_StellarFormationTime'][:] > 0 # bool
    star_id = f['PartType4/ParticleIDs'][star_select] # id
    star_subhalo_id = f['PartType4/SubhaloIDs'][star_select] # id
    star_pos = raytrace.unwrap(f['PartType4/Coordinates'][star_select] - com) # ckpc
    star_mass = f['PartType4/Masses'][star_select] * (1e10 * u.solMass).to(u.kg).value # kg


#
# Make a canvas and add simple view
#
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()


# create scatter object and fill in the data
gas_scatter = visuals.Markers()
gas_scatter.set_gl_state('translucent', depth_test=False)
gas_scatter.set_data(gas_pos, edge_width=0, face_color=(.7, .9, .8, .2), size=3*(gas_mass/np.std(gas_mass))**(0.3))

star_scatter = visuals.Markers()
star_scatter.set_gl_state('translucent', depth_test=False)
star_scatter.set_data(star_pos, edge_width=0, face_color=(1, .9, .8, .2), size=3*(star_mass/np.std(star_mass))**(0.3))

sfr_scatter = visuals.Markers()
sfr_scatter.set_gl_state('translucent', depth_test=False)
sfr_scatter.set_data(gas_pos, edge_width=0, face_color=(.2, .8, 1, .2), size=3*(gas_sfr/np.std(gas_sfr))**(0.3))

view.add(gas_scatter)
view.add(sfr_scatter)
view.add(star_scatter)

view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

@canvas.events.key_press.connect
def on_key_press(event):
    if event.text == 'q':
        sys.exit()

if __name__ == '__main__':
    if sys.flags.interactive != 1:
        vispy.app.run()
