import taichi as ti
import numpy as np
import time

# Initialize Taichi
ti.init(arch=ti.vulkan)

# Parameters
M = 100  # Grid size
N = 5    # Neighborhood size (NxN)
canvas_size = 800  # Canvas size for rendering
alpha = ti.field(dtype=ti.f32, shape=())  # Recovery rate
threshold = ti.field(dtype=ti.f32, shape=())  # Activation threshold
speed = ti.field(dtype=ti.f32, shape=())  # Speed of sum change

# Fields
state = ti.field(dtype=ti.i32, shape=(M, M))  # 0: off, 1: on
sum_field = ti.field(dtype=ti.f32, shape=(M, M))
weights = ti.field(dtype=ti.f32, shape=(M, M, N, N))  # Neighborhood weights
img = ti.Vector.field(3, dtype=ti.f32, shape=(M, M))  # Image field for rendering

# Initialize weights randomly such that they sum to 1
@ti.kernel
def initialize_weights():
    for i, j in ti.ndrange(M, M):
        total_weight = 0.0
        for di, dj in ti.ndrange(N, N):
            w = ti.random()
            weights[i, j, di, dj] = w
            total_weight += w
        # Normalize weights
        for di, dj in ti.ndrange(N, N):
            weights[i, j, di, dj] /= total_weight

# Update function
@ti.kernel
def update():
    for i, j in state:
        if sum_field[i, j] > 0:
            sum_field[i, j] -= speed[None]  # Decrease sum based on speed
        elif sum_field[i, j] < 0:
            sum_field[i, j] += speed[None]  # Recover sum based on speed

        if state[i, j] == 0 and sum_field[i, j] >= threshold[None]:
            state[i, j] = 1
            sum_field[i, j] = 0.0
        elif state[i, j] == 1:
            state[i, j] = 0
            sum_field[i, j] = -alpha[None]

    # Add contributions from neighbors
    for i, j in state:
        if sum_field[i, j] >= 0:
            for di, dj in ti.ndrange(N, N):
                ni = (i + di - N // 2) % M
                nj = (j + dj - N // 2) % M
                if state[ni, nj] == 1:
                    sum_field[i, j] += weights[i, j, di, dj]

# Kernel to update the image field based on the state
@ti.kernel
def update_img():
    for i, j in state:
        if state[i, j] == 1:
            img[i, j] = ti.Vector([1.0, 1.0, 1.0])
        else:
            img[i, j] = ti.Vector([0.0, 0.0, 0.0])

# Initialize fields and GUI
initialize_weights()
alpha[None] = 0.080
threshold[None] = 0.200
speed[None] = 0.010

stim_interval = 5
stim_timer = ti.field(dtype=ti.i32, shape=())
stim_timer[None] = stim_interval  # Initialize stim_timer

fps = 30
frame_duration = 1.0 / fps  # Time per frame

# Create Taichi window and canvas
window = ti.ui.Window("Cellular Automaton", (canvas_size, canvas_size), vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()

# Kernel to stimulate cells
@ti.kernel
def stimulate_cells(is_space_pressed: ti.i32):
    if is_space_pressed == 1:
        if stim_timer[None] < stim_interval:
            stim_timer[None] += 1
        elif stim_timer[None] == stim_interval:
            for i, j in ti.ndrange(M, M):
                if i % 10 == 0 and j % 10 == 0:
                    sum_field[i, j] = threshold[None] + 1.0  # Fire cell
            stim_timer[None] = 0

# Main simulation loop
while window.running:
    start_time = time.time()  # Record the start time of the frame

    # Handle user interaction for mouse clicks
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.LMB:
            mouse_pos = window.get_cursor_pos()
            x = int(mouse_pos[0] * M)
            y = int((1 - mouse_pos[1]) * M)  # Flip y-axis to match canvas
            if 0 <= x < M and 0 <= y < M:
                sum_field[x, M - y] = threshold[None] + 1.0  # Activate cell

    # Stimulate cells if space bar is pressed
    is_space_pressed = 1 if window.is_pressed(ti.ui.SPACE) else 0
    stimulate_cells(is_space_pressed)

    # Update simulation
    update()
    update_img()

    # Upscale the image to match the canvas size
    img_np = img.to_numpy()
    scale_factor = canvas_size // M
    img_resized = np.repeat(np.repeat(img_np, scale_factor, axis=0), scale_factor, axis=1)
    canvas.set_image(img_resized)

    # GUI for adjusting parameters
    with gui.sub_window("Settings", 0.05, 0.05, 0.3, 0.4):
        gui.text("Adjust Parameters")
        alpha_value = gui.slider_float("Alpha", alpha[None], 0.01, 0.5)
        threshold_value = gui.slider_float("Threshold", threshold[None], 0.1, 3.0)
        speed_value = gui.slider_float("Speed", speed[None], 0.001, 0.1)
        alpha[None] = alpha_value
        threshold[None] = threshold_value
        speed[None] = speed_value

    # Calculate remaining time to achieve desired FPS
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_duration:
        time.sleep(frame_duration - elapsed_time)

    window.show()
