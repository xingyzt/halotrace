import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy.visuals.transforms import MatrixTransform
import h5py
import time
import os
import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as const
import raytrace
import sys
import argparse
from numpy.random import default_rng

def parse_args():
    parser = argparse.ArgumentParser(description="Render halos in 3D")

    # Default integer (positional)
    parser.add_argument("halo_index", type=int, nargs="?", default=0, help="halo index in dm_proto (default: 0)")

    # Optional integers with flags
    parser.add_argument("--halo_id", type=int, help="halo ID (optional, overrides index)")
    parser.add_argument("--max_particles", type=int, default=6,
                        help="log10 maximum number of particles (default: 6)")

    # Boolean flag (store_true means False by default, True if passed)
    parser.add_argument("--render", action="store_true",
                        help="rendering to webp (default: False)")

    parser.add_argument("--force", action="store_true",
                        help="overwrite old renders (default: False)")

    args = parser.parse_args()
    return args

with h5py.File("outputs/tng_cluster_catalog.hdf5", "r") as f:
    halo_ids = f['table/haloID'][:]

if __name__ == "__main__":
    args = parse_args()

    max_particles = int(10**args.max_particles)

    halo_id = args.halo_id
    halo_index = args.halo_index
    if halo_id == None:
        halo_id = halo_ids[args.halo_index]
    else:
        halo_index = np.where(halo_ids == halo_index)[0,0]
    print(halo_index, halo_id)

    if args.render:
        out = f'vis/3d_gl/halo_{halo_id}.webp'
        if os.path.exists(out) and not args.force:
            sys.exit()

        import imageio
        import tqdm


with h5py.File("outputs/tng_cluster_catalog.hdf5", "r") as f:

    print(f['table/GroupMassType'][halo_index])
    com = f['table/GroupCM'][halo_index, :]
    center = f['table/GroupPos'][halo_index, :]
    r500 = f['table/Group_R_Crit500'][halo_index]
    r200 = f['table/Group_R_Crit200'][halo_index]
    m500 = f['table/mhalo_500c'][halo_index]
    f500 = f['table/fgas_r500'][halo_index]

    axis = com - center
    relax = np.linalg.norm(axis)
    axis /= relax
    x = np.array([1,0,0])
    transform = MatrixTransform()
    transform.rotate(np.degrees(np.acos(np.dot(axis, x))), np.cross(axis, x))
   #subhalo_id = f['subhalos']["id"][:]

with h5py.File(f'tng_cache/snap_099/cutout_{halo_id}.hdf5', 'r') as f:

    gas_count = f['PartType0/ParticleIDs'][:].size
    gas_multiplicity = 1.0
    gas_select = np.ones(gas_count) > 0
    if gas_count > max_particles:
        gas_multiplicity = float(gas_count)/float(max_particles)
        gas_select = sorted(default_rng().choice(range(gas_count), size=max_particles, replace=False))

    gas_mass = f['PartType0/Masses'][:][gas_select] * ((1e10 * u.solMass).to(u.kg) / cu.littleh) # kg ckpc^-3
    gas_density = f['PartType0/Density'][:][gas_select] * (1e10 * u.solMass).to(u.kg) / cu.littleh / ( u.kpc / cu.littleh)**3 # kg ckpc^-3
    gas_volume = gas_mass/gas_density
    gas_radius = ((0.75/np.pi)*gas_volume)**(1/3)
    
    gas_pos = raytrace.unwrap(f['PartType0/Coordinates'][:][gas_select] - center) # ckpc
   #gas_electron_abundance = f['PartType0/ElectronAbundance'][:][gas_select] # fraction
   #gas_hydrogen_massfrac = f['PartType0/GFM_Metals'][:][gas_select,0] # fraction
   #gas_id = f['PartType0/ParticleIDs'][:][gas_select] # id
   #gas_subhalo_id = f['PartType0/SubhaloIDs'][:][gas_select] # id
   #gas_sfr = f['PartType0/StarFormationRate'][:][gas_select] # Msun/yr
   #gas_n_e = (gas_density * gas_electron_abundance * gas_hydrogen_massfrac / const.m_p).value # ckpc^-3
   #gas_N_e = (gas_mass * gas_electron_abundance * gas_hydrogen_massfrac / const.m_p).value # ckpc^-3

    # separately count the SFR stuff
    gas_sfr_select = f['PartType0/StarFormationRate'][:] > 0
    gas_sfr = f['PartType0/StarFormationRate'][:][gas_sfr_select] # Msun/year
    gas_sfr_pos = raytrace.unwrap(f['PartType0/Coordinates'][:][gas_sfr_select] - center) # ckpc
    gas_sfr_mass = f['PartType0/Masses'][:][gas_sfr_select] * ((1e10 * u.solMass).to(u.kg) / cu.littleh) # kg ckpc^-3
    gas_sfr_density = f['PartType0/Density'][:][gas_sfr_select] * (1e10 * u.solMass).to(u.kg) / cu.littleh / ( u.kpc / cu.littleh)**3 # kg ckpc^-3
    gas_sfr_volume = gas_sfr_mass/gas_sfr_density
    gas_sfr_radius = ((0.75/np.pi)*gas_sfr_volume)**(1/3)

    star_select = f['PartType4/GFM_StellarFormationTime'][:] > 0
    star_notwind = np.nonzero(star_select)[0] # indices of stars that are not wind particles
    star_count = star_notwind.size
    star_multiplicity = 1.0
    if star_count > max_particles:
        star_multiplicity = float(star_count)/float(max_particles)
        star_select = sorted(default_rng().choice(star_notwind, size=max_particles, replace=False))
    
   #star_id = f['PartType4/ParticleIDs'][:][star_select] # id
   #star_subhalo_id = f['PartType4/SubhaloIDs'][:][star_select] # id
    star_pos = raytrace.unwrap(f['PartType4/Coordinates'][:][star_select] - center) # ckpc
    star_mass = f['PartType4/Masses'][:][star_select] * (1e10 * u.solMass).to(u.kg).value # kg


#
# Make a canvas and add simple view
#
canvas = vispy.scene.SceneCanvas(keys='interactive', show=(not args.render), size = (1000, 600))
view = canvas.central_widget.add_view()


# create scatter object and fill in the data
if gas_pos.size > 0:
    gas_scatter = visuals.Markers(scaling='scene')
    gas_scatter.transform = transform
    gas_scatter.set_gl_state('additive', depth_test=False)
    ones = np.ones_like(gas_density)
    gas_scatter.set_data(gas_pos, edge_width=0, face_color=np.array((0.*ones, .0*ones, 1.*ones, (gas_density/np.max(gas_density))**0.33)).T, size=(gas_multiplicity**0.33)*gas_radius)
    view.add(gas_scatter)

if star_pos.size > 0:
    star_scatter = visuals.Markers(scaling='scene')
    star_scatter.transform = transform
    star_scatter.set_gl_state('additive', depth_test=False)
    star_scatter.set_data(star_pos, edge_width=0, face_color=(1., .0, .0, .5), size=(star_multiplicity*star_mass/np.std(star_mass))**0.33)
    view.add(star_scatter)

if gas_sfr_pos.size > 0:
    sfr_scatter = visuals.Markers(scaling='scene')
    sfr_scatter.transform = transform
    sfr_scatter.set_gl_state('additive', depth_test=False)
    ones = np.ones_like(gas_sfr_density)
    sfr_scatter.set_data(gas_sfr_pos, edge_width=0, face_color=(.5, 1., .1, .5), size=(10*gas_sfr)**0.33)
    view.add(sfr_scatter)

circle500 = visuals.Ellipse(radius=(r500,r500), center=(0,0,0), color=None, border_color='red')
circle500.set_gl_state('additive', depth_test=False)
view.add(circle500)

circle200 = visuals.Ellipse(radius=(r200,r200), center=(0,0,0), color=None, border_color='blue')
circle200.set_gl_state('additive', depth_test=False)
view.add(circle200)

grid = visuals.GridLines(scale=(1,1), grid_bounds=((-500,500),(-500,500))) # 1 Mpc box
view.add(grid)

text = visuals.Text(f'#{halo_index}/{halo_id}, @500c: 10^{m500:.1f} Msun, {r500:.0f} kpc, {100*f500:.0f}% gas', 
                    face='Unifont', font_size=16, parent=canvas.scene, color='#0f0', pos=(canvas.size[0]//2, 20))
# view.add(text)

view.camera = 'turntable'  # or try 'arcball'
view.camera.distance = 2e3 # 2 Mpc
view.camera.fov = 70.
view.camera.elevation = 20.
view.camera.center = (0,0,0)

@canvas.events.key_press.connect
def on_key_press(event):
    if event.text == 'q':
        sys.exit()

if args.render:
    writer = imageio.get_writer(out, loop=0, fps=16, quality=99)

    n_steps = 120
    step_angle = 360 / n_steps
    for i in tqdm.trange(n_steps):
        view.camera.orbit(step_angle, 0)
        im = canvas.render(alpha=False)
        writer.append_data(im)
    writer.close()
elif __name__ == '__main__':
    vispy.app.run()
