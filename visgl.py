import numpy as np
import vispy.scene
from vispy.scene import visuals
import h5py
import os
import sys
import argparse
from numpy.random import default_rng
from vispy import app, gloo, scene, io
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
    parser.add_argument("--gamma", type=float, default=4.0,
            help="gamma correction value (default: 4.0)")
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
        out = lambda i: f"vis/3d_gl/halo_{halo_id}_{i:003}.png"
        if os.path.exists(out(0)) and not args.force:
            print("Output exists. Use --force to overwrite.")
            sys.exit()
       #import imageio
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
            size=(1920, 1080), bgcolor='black')
    canvas._draw_scene = lambda *args, **kwargs: None
    view = canvas.central_widget.add_view()

    width, height = canvas.size
    tex_depth = gloo.RenderBuffer((height, width), format='depth')

    # 3 color textures (8-bit RGB)
    tex_gas = gloo.Texture2D((height, width, 4), internalformat='rgba8')
    tex_star = gloo.Texture2D((height, width, 4), internalformat='rgba8')
    tex_sfr = gloo.Texture2D((height, width, 4), internalformat='rgba8')

    # Corresponding framebuffers
    fbo_gas  = gloo.FrameBuffer(tex_gas,  tex_depth)
    fbo_star = gloo.FrameBuffer(tex_star, tex_depth)
    fbo_sfr  = gloo.FrameBuffer(tex_sfr,  tex_depth)

    symbol = 'diamond'

    vis = {
            'gas0': visuals.Markers(),
            'gas1': visuals.Markers(),
            'gas2': visuals.Markers(),
            'gas3': visuals.Markers(),
            'gas_star': visuals.Markers(),
            'gas_sfr': visuals.Markers(),
            'star': visuals.Markers(),
            'sfr': visuals.Markers()
            }
    for k in vis:
        vis[k].scaling = 'scene'
        vis[k].set_gl_state('additive', depth_test=False, blend_func=('one', 'one'))
        view.add(vis[k])

    if gas_pos.size > 0:
        val = gas_surface_density / 3e8 
        nbits = 1.2

        zeros = np.zeros_like(val)

        gas_color = np.clip(np.stack([val, val * nbits, val * nbits**2, val * nbits**3], axis=1), 0, 1) 
        gas_r_eff = (gas_multiplicity ** 0.33) * gas_radius
        vis['gas0'].set_data(gas_pos, edge_width=0, face_color=gas_color, size=gas_r_eff, symbol=symbol)

        gas_color = np.clip(np.stack([val * nbits**4, val * nbits ** 5, val * nbits**6, val * nbits**7], axis=1), 0, 1) 
        vis['gas1'].set_data(gas_pos, edge_width=0, face_color=gas_color, size=gas_r_eff, symbol=symbol)

        gas_color = np.clip(np.stack([val * nbits ** 8, val * nbits**9, val * nbits**10, val*nbits**11 ], axis=1), 0, 1) 
        vis['gas2'].set_data(gas_pos, edge_width=0, face_color=gas_color, size=gas_r_eff, symbol=symbol)

        gas_color = np.clip(np.stack([val * nbits**12, val * nbits ** 13, val * nbits**14, val * nbits**15], axis=1), 0, 1) 
        vis['gas3'].set_data(gas_pos, edge_width=0, face_color=gas_color, size=gas_r_eff, symbol=symbol)

        gas_color = np.clip(np.stack([val * nbits**16, val * nbits ** 17, val * nbits**18, zeros], axis=1), 0, 1) 
        vis['gas_star'].set_data(gas_pos, edge_width=0, face_color=gas_color, size=gas_r_eff, symbol=symbol)

        gas_color = np.clip(np.stack([val * nbits**19, val * nbits ** 20, val * nbits**21, zeros], axis=1), 0, 1) 
        vis['gas_sfr'].set_data(gas_pos, edge_width=0, face_color=gas_color, size=gas_r_eff, symbol=symbol)


    if star_pos.size > 0:
        vis['star'].set_data(star_pos, edge_width=0, face_color=(0, 0, 0, 0.01),
                              size=(star_multiplicity * star_mass / 1e6) ** 0.33, symbol=symbol)

    if gas_sfr_pos.size > 0:
        vis['sfr'].set_data(gas_sfr_pos, edge_width=0, face_color=(0, 0, 0, 0.05),
                             size=(1e5 * gas_sfr) ** 0.33, symbol=symbol)

    # Add circles
    circs = list()
    for r in (r500, r200):
        circ = visuals.Ellipse(radius=(r, r), center=(0, 0, 0),
                               color=None, border_color=(0,0,0,0.7), num_segments=16)
        circ.set_gl_state('additive', depth_test=False) 
        circs.append(circ)
        view.add(circ)

    grids = list()
    for i in range(1, 33): 
        c = (0,0,0,0.7/i**0.3) 
        r = i*1000 
        grid = visuals.Ellipse(radius=(r,r), center=(0,0,0), color=None, border_color=c, num_segments=64)
        grid.set_gl_state('additive', depth_test=False) 
        grids.append(grid)
        view.add(grid)

    view.camera = 'turntable'
    view.camera.distance = 2e3
    view.camera.fov = 70.
    view.camera.elevation = 20.
    view.camera.center = (0, 0, 0)

    # Info text
    halo_text = (f"#{halo_index}:{halo_id}, "
                 f"@500c: 10^{m500:.1f} Msun, "
                 f"{r500:.0f} kpc, {100*f500:.1f}% gas")
    text = visuals.Text(halo_text, color=(0,0,0,1), font_size=16, face='Nimbus Mono PS', bold=True,
                        anchor_x="center", anchor_y="bottom", parent=canvas.scene)
    text.pos = (canvas.size[0] / 2, 24)

    @canvas.events.key_press.connect
    def on_key_press(event):
        if event.text == 'q':
            app.quit()

    def render():
        # Pass 1 – render each group into its own framebuffer
        with fbo_gas:
            gloo.clear(color=True, depth=True)
            vis['gas0'].draw()
            arr_gas0  = np.float64(fbo_gas.read(mode='color')) / 255.
            gloo.clear(color=True, depth=True)
            vis['gas1'].draw()
            arr_gas1  = np.float64(fbo_gas.read(mode='color')) / 255.
            gloo.clear(color=True, depth=True)
            vis['gas2'].draw()
            arr_gas2  = np.float64(fbo_gas.read(mode='color')) / 255.
            gloo.clear(color=True, depth=True)
            vis['gas3'].draw()
            arr_gas3  = np.float64(fbo_gas.read(mode='color')) / 255.
        with fbo_star:
            gloo.clear(color=True, depth=True)
            vis['star'].draw()
            vis['gas_star'].draw()
            for grid in grids:
                grid.draw()
            arr_star = np.float64(fbo_star.read(mode='color')) / 255.
        with fbo_sfr:
            gloo.clear(color=True, depth=True)
            vis['sfr'].draw()
            vis['gas_sfr'].draw()
            for circ in circs:
                circ.draw()
            text.draw()
            arr_sfr  = np.float64(fbo_sfr.read(mode='color')) / 255.


        p = 1/args.gamma

        total_gas = np.concatenate([ arr_gas0, arr_gas1, arr_gas2, arr_gas3, arr_star[:,:,:-1], arr_sfr[:,:,:-1] ], axis=-1)
        n_channels = total_gas.shape[-1]
        sum_gas = total_gas[:,:,-1]
        for i in range(n_channels - 1, -1, -1):
            sum_gas *= float(1.0/nbits)
            sum_gas += np.clip(total_gas[:,:,i] - 1.0/nbits, 0, 1)

        im = np.zeros((total_gas.shape[0], total_gas.shape[1], 3), dtype='f8')
        im[:,:,0] = arr_star[:,:,-1]
        im[:,:,1] = arr_sfr[:,:,-1]
        im[:,:,2] = np.clip(np.log10(sum_gas + 1e-100)/6 + 1, 0, 1)**3
        return np.uint8(im*255.)

    @canvas.connect
    def on_draw(event):
        render()


   #    # Pass 2 – combine to screen
   #    gloo.clear(color=True, depth=True)
   #    post_program['tex_gas']  = tex_gas
   #    post_program['tex_star'] = tex_star
   #    post_program['tex_sfr']  = tex_sfr
   #    post_program.draw('triangle_strip')

    # -------------------------------------------------
    # Render to file (optional)
    # -------------------------------------------------
    if args.render:
        # writer = imageio.get_writer(out, fps=16, codec='libwebp', quality=100)
        n_steps = 360
        for i in trange(n_steps):
            view.camera.orbit(360 / n_steps, 0)
            canvas.render()
            im = render()
            io.write_png(out(i), im)
            # writer.append_data(im)
        # writer.close()
        print(f"Saved {out(i)}")
    else:
        app.run()

