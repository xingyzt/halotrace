import numpy as np
import vispy.scene
from vispy.scene import visuals
import h5py
import os
import sys
import argparse
from numpy.random import default_rng
from vispy import app, gloo, scene
import raytrace


# -------------------------------------------------
# CLI argument parsing
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Render halos in 3D with gamma correction")
    parser.add_argument("halo_index", type=int, nargs="?", default=0,
                        help="halo index in dm_proto (default: 0)")
    parser.add_argument("--halo_id", type=int, help="halo ID (optional, overrides index)")
    parser.add_argument("--max_particles", type=float, default=6.0,
                        help="log10 maximum number of particles (default: 6.0)")
    parser.add_argument("--gamma", type=float, default=2.2,
                        help="gamma correction value (default: 2.2)")
    parser.add_argument("--render", action="store_true", help="rendering to webp (default: False)")
    parser.add_argument("--force", action="store_true", help="overwrite old renders (default: False)")
    return parser.parse_args()


# -------------------------------------------------
# Load halo list
# -------------------------------------------------
with h5py.File("outputs/tng_cluster_catalog.hdf5", "r") as f:
    halo_ids = f['table/haloID'][:]


# -------------------------------------------------
# Main logic
# -------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    max_particles = int(10 ** args.max_particles)

    # Choose halo by ID or index
    if args.halo_id is None:
        halo_id = halo_ids[args.halo_index]
        halo_index = args.halo_index
    else:
        halo_id = args.halo_id
        match = np.where(halo_ids == halo_id)[0]
        if len(match) == 0:
            raise ValueError(f"Halo ID {halo_id} not found.")
        halo_index = match[0]

    print("Using halo:", halo_index, halo_id)

    if args.render:
        out = f"vis/3d_gl/halo_{halo_id}.webp"
        if os.path.exists(out) and not args.force:
            print("Output exists. Use --force to overwrite.")
            sys.exit()
        import imageio
        from tqdm import trange

    # -------------------------------------------------
    # Read halo properties
    # -------------------------------------------------
    with h5py.File("outputs/tng_cluster_catalog.hdf5", "r") as f:
        com = f['table/GroupCM'][halo_index, :]
        center = f['table/GroupPos'][halo_index, :]
        r500 = f['table/Group_R_Crit500'][halo_index]
        r200 = f['table/Group_R_Crit200'][halo_index]
        m500 = f['table/mhalo_500c'][halo_index]
        f500 = f['table/fgas_r500'][halo_index]

    # -------------------------------------------------
    # Load particle data
    # -------------------------------------------------
    with h5py.File(f"tng_cache/snap_099/cutout_{halo_id}.hdf5", "r") as f:
        gas_count = f['PartType0/ParticleIDs'][:].size
        gas_select = np.ones(gas_count, dtype=bool)
        gas_multiplicity = 1.0
        if gas_count > max_particles:
            gas_select = sorted(default_rng().choice(range(gas_count), size=max_particles, replace=False))
            gas_multiplicity = float(gas_count) / float(max_particles)

        gas_mass = f['PartType0/Masses'][:][gas_select] * 1e10
        gas_density = f['PartType0/Density'][:][gas_select] * 1e10
        gas_volume = gas_mass / gas_density
        gas_radius = ((0.75 / np.pi) * gas_volume) ** (1 / 3)
        gas_surface_density = gas_mass / (np.pi * gas_radius ** 2)
        gas_pos = raytrace.unwrap(f['PartType0/Coordinates'][:][gas_select] - center)
            

        # SFR particles
        gas_sfr_select = f['PartType0/StarFormationRate'][:] > 0
        gas_sfr = f['PartType0/StarFormationRate'][:][gas_sfr_select]
        gas_sfr_pos = raytrace.unwrap(f['PartType0/Coordinates'][:][gas_sfr_select] - center)
        gas_sfr_mass = f['PartType0/Masses'][:][gas_sfr_select] * 1e10
        gas_sfr_density = f['PartType0/Density'][:][gas_sfr_select] * 1e10
        gas_sfr_volume = gas_sfr_mass / gas_sfr_density
        gas_sfr_radius = ((0.75 / np.pi) * gas_sfr_volume) ** (1 / 3)
        gas_sfr_surface_density = gas_sfr_mass / (np.pi * gas_sfr_radius ** 2)

        star_mask = f['PartType4/GFM_StellarFormationTime'][:] > 0
        star_count = star_mask.size
        star_select = np.nonzero(star_mask)[0]
        star_multiplicity = 1.0
        if star_count > max_particles:
            star_select = sorted(default_rng().choice(star_select, size=max_particles, replace=False))
            star_multiplicity = float(star_count) / float(max_particles)

        star_pos = raytrace.unwrap(f['PartType4/Coordinates'][:][star_select] - center)
        star_mass = f['PartType4/Masses'][:][star_select] * 1e10

    # -------------------------------------------------
    # Setup canvas + view
    # -------------------------------------------------
    canvas = scene.SceneCanvas(keys='interactive', show=(not args.render),
                               size=(1600, 900), bgcolor='black')
    view = canvas.central_widget.add_view()
    symbol = 'diamond'

    if gas_pos.size > 0:
        gas_scatter = visuals.Markers(scaling='scene')
        gas_scatter.set_gl_state('additive', depth_test=False)
        ones = np.ones_like(gas_density)
        zeros = np.zeros_like(gas_density)
        alpha = np.clip(gas_surface_density / 1e7, 0, 1)
        gas_color = np.stack([zeros, zeros, alpha, ones], axis=1)
        gas_scatter.set_data(gas_pos, edge_width=0, face_color=gas_color,
                             size=(gas_multiplicity ** 0.33) * gas_radius, symbol=symbol)
        view.add(gas_scatter)

    if star_pos.size > 0:
        star_scatter = visuals.Markers(scaling='scene')
        star_scatter.set_gl_state('additive', depth_test=False)
        star_scatter.set_data(star_pos, edge_width=0, face_color=(0.01, 0, 0, 1),
                              size=(star_multiplicity * star_mass / 1e6) ** 0.33, symbol=symbol)
        view.add(star_scatter)

    if gas_sfr_pos.size > 0:
        sfr_scatter = visuals.Markers(scaling='scene')
        sfr_scatter.set_gl_state('additive', depth_test=False)
        sfr_scatter.set_data(gas_sfr_pos, edge_width=0, face_color=(0, 0.05, 0, 1),
                             size=(1e5 * gas_sfr) ** 0.33, symbol=symbol)
        view.add(sfr_scatter)

    # Add circles
    for radius, color in [(r500, 'red'), (r200, 'blue')]:
        circ = visuals.Ellipse(radius=(radius, radius), center=(0, 0, 0),
                               color=None, border_color=color, num_segments=64)
        circ.set_gl_state('additive', depth_test=False)
        view.add(circ)

    for i in range(1, 33): 
        color=(0,1,0,0.5/i**0.5) 
        r = i*1000 
        grid = visuals.Ellipse(radius=(r,r), center=(0,0,0), color=None, border_color=color, num_segments=8) 
        grid.set_gl_state('additive', depth_test=False) 
        view.add(grid)

    view.camera = 'turntable'
    view.camera.distance = 2e3
    view.camera.fov = 70.
    view.camera.elevation = 20.
    view.camera.center = (0, 0, 0)

    # Info text
    halo_text = (f"#{halo_index}/{halo_id}, "
                 f"@500c: 10^{m500:.1f} Msun, "
                 f"{r500:.0f} kpc, {100*f500:.0f}% gas")
    text = visuals.Text(halo_text, color="#0f0", font_size=16, face='Unifont',
                        anchor_x="center", anchor_y="bottom", parent=canvas.scene)
    text.pos = (canvas.size[0] / 2, 24)

    @canvas.events.key_press.connect
    def on_key_press(event):
        if event.text == 'q':
            app.quit()


    # -------------------------------------------------
    # Render to file (optional)
    # -------------------------------------------------
    if args.render:
        writer = imageio.get_writer(out, fps=16, codec='libwebp', quality=100)
        n_steps = 120
        for i in trange(n_steps):
            view.camera.orbit(360 / n_steps, 0)
            im = canvas.render()
            writer.append_data(im)
        writer.close()
        print(f"Saved {out}")
    else:
        app.run()

