from PIL import Image, ImageSequence
import os
import math

# Path to your directory of gifs
input_dir = "vis/3d_gl/"
output_file = "mergedgrid.gif"

# --- Load all gifs ---
gif_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".gif")]
gif_files.sort()  # Or random.shuffle for random arrangement
gif_files = gif_files[:100]

# --- Extract frames from each gif ---
all_gifs = []
max_frames = 0
for f in gif_files:
    im = Image.open(os.path.join(input_dir, f))
    frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(im)]
    all_gifs.append(frames)
    max_frames = max(max_frames, len(frames))

# --- Normalize frame count (loop shorter gifs) ---
for i in range(len(all_gifs)):
    frames = all_gifs[i]
    if len(frames) < max_frames:
        extended = []
        for j in range(max_frames):
            extended.append(frames[j % len(frames)])
        all_gifs[i] = extended

# --- Make sure all frames have same size ---
w, h = all_gifs[0][0].size
for i in range(len(all_gifs)):
    all_gifs[i] = [frame.resize((w, h), Image.LANCZOS) for frame in all_gifs[i]]

# --- Build grid frames ---
grid_size = 9
grid_w, grid_h = grid_size * w, grid_size * h
grid_frames = []

for frame_idx in range(max_frames):
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    for gif_idx, frames in enumerate(all_gifs):
        row, col = divmod(gif_idx, grid_size)
        canvas.paste(frames[frame_idx], (col * w, row * h), frames[frame_idx])
    grid_frames.append(canvas)

# --- Save as animated gif ---
grid_frames[0].save(
    output_file,
    save_all=True,
    append_images=grid_frames[1:],
    loop=0,
    duration=60,   # ms per frame (adjust as needed)
    disposal=2
)

print(f"Saved animated grid gif as {output_file}")

