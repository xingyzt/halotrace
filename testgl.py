from vispy import app, gloo, scene
from vispy.scene import visuals
import numpy as np

canvas = scene.SceneCanvas(keys='interactive', show=True, size=(1280, 720), bgcolor='black')
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

# Example: three random point clouds
def random_cloud(n, color):
    pts = np.random.uniform(-1, 1, (n, 3))
    col = np.tile(color, (n, 1))
    return pts * 300, col

pts_gas, col_gas = random_cloud(2000, (0.0, 0.6, 1.0, 0.6))
pts_star, col_star = random_cloud(1500, (1.0, 0.8, 0.0, 0.6))
pts_sfr, col_sfr = random_cloud(1000, (0.2, 1.0, 0.2, 0.6))

scat_gas = visuals.Markers()
scat_star = visuals.Markers()
scat_sfr = visuals.Markers()
for s, p, c in [(scat_gas, pts_gas, col_gas),
                (scat_star, pts_star, col_star),
                (scat_sfr, pts_sfr, col_sfr)]:
    s.set_data(p, face_color=c, size=6)
    s.set_gl_state('additive', depth_test=False)
    view.add(s)

view.camera = 'turntable'
view.camera.distance = 1000

# --- Postprocess shader for compositing ---
vertex_shader = """
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
    v_uv = 0.5 * (a_pos + 1.0);
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

fragment_shader = """
uniform sampler2D tex_gas;
uniform sampler2D tex_star;
uniform sampler2D tex_sfr;
uniform float gamma;
varying vec2 v_uv;
void main() {
    vec3 g = texture2D(tex_gas,  v_uv).rgb;
    vec3 s = texture2D(tex_star, v_uv).rgb;
    vec3 f = texture2D(tex_sfr,  v_uv).rgb;
    vec3 col = g - s + f;        // add together
    col = pow(col, vec3(1.0 / gamma)); // gamma correction
    gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
}
"""

post_program = gloo.Program(vertex_shader, fragment_shader)
quad = np.array([[-1,-1],[1,-1],[-1,1],[1,1]], np.float32)
post_program['a_pos'] = quad
post_program['gamma'] = 2.2


@canvas.connect
def on_draw(event):
    # Pass 1 – render each group into its own framebuffer
    with fbo_gas:
        gloo.clear(color=True, depth=True)
        scat_gas.draw()
    with fbo_star:
        gloo.clear(color=True, depth=True)
        scat_star.draw()
    with fbo_sfr:
        gloo.clear(color=True, depth=True)
        scat_sfr.draw()

    # Pass 2 – combine to screen
    gloo.clear(color=True, depth=True)
    post_program['tex_gas']  = tex_gas
    post_program['tex_star'] = tex_star
    post_program['tex_sfr']  = tex_sfr
    post_program.draw('triangle_strip')

app.run()

