import taichi as ti
import numpy as np
import time

# Initialize Taichi
ti.init(arch=ti.vulkan)

# Parameters
M = 100  # Grid size
N = 5    # Neighborhood size (NxN)
canvas_size = 800  # Canvas size for rendering

# Fields for hyperparameters
alpha = ti.field(dtype=ti.f32, shape=())  # Recovery rate
speed = ti.field(dtype=ti.f32, shape=())  # Speed of sum change
learning_rate = ti.field(dtype=ti.f32, shape=())  # Learning rate for Hebbian learning

# Homeostatic Plasticity Parameters
target_activity = ti.field(dtype=ti.f32, shape=())  # Target average activity per cell
threshold_learning_rate = ti.field(dtype=ti.f32, shape=())  # Learning rate for threshold adjustments

# Fields
state = ti.field(dtype=ti.i32, shape=(M, M))       # 0: off, 1: on
sum_field = ti.field(dtype=ti.f32, shape=(M, M))   # Sum of inputs
weights = ti.field(dtype=ti.f32, shape=(M, M, N, N))  # Neighborhood weights
threshold = ti.field(dtype=ti.f32, shape=(M, M))   # Per-cell threshold
activity = ti.field(dtype=ti.f32, shape=(M, M))    # Average activity per cell
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

# Initialize per-cell thresholds with random deviations
@ti.kernel
def initialize_thresholds():
    for i, j in ti.ndrange(M, M):
        threshold[i, j] = 0.3 + (2.0*ti.random(ti.f32) - 1.0) * 0.1  # Deviation between -0.1 and +0.1

# Update function
@ti.kernel
def update():
    for i, j in state:
        if sum_field[i, j] > 0:
            sum_field[i, j] -= speed[None]  # Decrease sum based on speed
        elif sum_field[i, j] < 0:
            sum_field[i, j] += speed[None]  # Recover sum based on speed

        if state[i, j] == 0 and sum_field[i, j] >= threshold[i, j]:
            state[i, j] = 1
            sum_field[i, j] = 0.0
        elif state[i, j] == 1:
            state[i, j] = 0
            sum_field[i, j] = -alpha[None]

    # Add contributions from neighbors
    for i, j in state:
        if sum_field[i, j] >= 0:
            for di, dj in ti.ndrange(N, N):
                ni = i + di - N // 2
                nj = j + dj - N // 2
                if 0 <= ni < M and 0 <= nj < M:
                    if state[ni, nj] == 1:
                        sum_field[i, j] += weights[i, j, di, dj]

# Hebbian learning kernel
@ti.kernel
def hebbian_learning():
    for i, j in ti.ndrange(M, M):
        total_weight = 0.0
        for di, dj in ti.ndrange(N, N):
            ni = i + di - N // 2
            nj = j + dj - N // 2
            if 0 <= ni < M and 0 <= nj < M:
                if state[i, j] == 1 and state[ni, nj] == 1:
                    weights[i, j, di, dj] += learning_rate[None]
                # Ensure weights are non-negative
                if weights[i, j, di, dj] < 0:
                    weights[i, j, di, dj] = 0.0
                total_weight += weights[i, j, di, dj]
        # Normalize weights
        if total_weight > 0:
            for di, dj in ti.ndrange(N, N):
                weights[i, j, di, dj] /= total_weight

# Homeostatic plasticity kernel
@ti.kernel
def homeostatic_plasticity():
    decay = 0.99  # Decay factor for exponential moving average
    for i, j in ti.ndrange(M, M):
        # Update the exponential moving average of activity
        activity[i, j] = decay * activity[i, j] + (1 - decay) * float(state[i, j])

        # Adjust threshold based on average activity
        if activity[i, j] > target_activity[None]:
            threshold[i, j] += threshold_learning_rate[None]
        elif activity[i, j] < target_activity[None]:
            threshold[i, j] -= threshold_learning_rate[None]

        # Ensure thresholds stay within reasonable bounds
        threshold[i, j] = ti.max(0.05, ti.min(threshold[i, j], 1.0))

# Kernel to update the image field based on the state
@ti.kernel
def update_img():
    for i, j in state:
        if state[i, j] == 1:
            img[i, j] = ti.Vector([1.0, 1.0, 1.0])
        else:
            img[i, j] = ti.Vector([0.0, 0.0, 0.0])

# Initialize fields
initialize_weights()
initialize_thresholds()

# Hyperparameters
alpha[None] = 0.040
speed[None] = 0.010
learning_rate[None] = 0.01

# Homeostatic Plasticity Parameters
target_activity[None] = 0.1  # Target activity level (between 0 and 1)
threshold_learning_rate[None] = 0.008  # Learning rate for threshold adjustment

# Stimulation parameters
stim_interval = 5
stim_timer = ti.field(dtype=ti.i32, shape=())
stim_timer[None] = stim_interval  # Initialize stim_timer
random_stimulation = ti.field(dtype=ti.i32, shape=())  # Toggle for random stimulation
random_stimulation[None] = 0  # Default to off

fps = 30
frame_duration = 1.0 / fps  # Time per frame

# Create Taichi window and canvas
window = ti.ui.Window("Cellular Automaton with Homeostatic Plasticity", (canvas_size, canvas_size), vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()

# Fields for stimulation interval and lattice size
stim_interval_value = ti.field(dtype=ti.i32, shape=())
stim_lattice_size = ti.field(dtype=ti.i32, shape=())
stim_interval_value[None] = stim_interval
stim_lattice_size[None] = 10  # Default lattice size for stimulation

# Kernel to stimulate cells
@ti.kernel
def stimulate_cells(is_space_pressed: ti.i32):
    if is_space_pressed == 1:
        if stim_timer[None] < stim_interval_value[None]:
            stim_timer[None] += 1
        elif stim_timer[None] == stim_interval_value[None]:
            if random_stimulation[None] == 1:  # Random stimulation
                for i, j in ti.ndrange(M, M):
                    if ti.random() < 0.1:  # 10% chance to stimulate a cell
                        sum_field[i, j] = threshold[i, j] + 1.0
            else:  # Lattice-based stimulation
                for i, j in ti.ndrange(M, M):
                    if i % stim_lattice_size[None] == 0 and j % stim_lattice_size[None] == 0:
                        if i < M // 6:
                            sum_field[i, j] = threshold[i, j] + 1.0
            stim_timer[None] = 0
    else:
        stim_timer[None] = 0

# Main simulation loop with additional GUI sliders
while window.running:
    start_time = time.time()  # Record the start time of the frame

    # Handle user interaction for mouse clicks
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.LMB:
            mouse_pos = window.get_cursor_pos()
            x = int(mouse_pos[0] * M)
            y = int((1 - mouse_pos[1]) * M)  # Flip y-axis to match canvas
            if 0 <= x < M and 0 <= y < M:
                sum_field[x, M - y] = threshold[x, M - y] + 1.0  # Activate cell

    # Stimulate cells if space bar is pressed
    is_space_pressed = 1 if window.is_pressed(ti.ui.SPACE) else 0
    stimulate_cells(is_space_pressed)

    # Update simulation
    update()
    hebbian_learning()
    homeostatic_plasticity()
    update_img()

    # Upscale the image to match the canvas size
    img_np = img.to_numpy()
    scale_factor = canvas_size // M
    img_resized = np.repeat(np.repeat(img_np, scale_factor, axis=0), scale_factor, axis=1)
    canvas.set_image(img_resized)

    # GUI for adjusting parameters
    with gui.sub_window("Settings", 0.05, 0.05, 0.4, 0.6):
        gui.text("Adjust Parameters")
        alpha_value = gui.slider_float("Alpha", alpha[None], 0.01, 0.5)
        speed_value = gui.slider_float("Speed", speed[None], 0.001, 0.1)
        learning_rate_value = gui.slider_float("Hebbian Learning Rate", learning_rate[None], 0.0, 0.1)
        alpha[None] = alpha_value
        speed[None] = speed_value
        learning_rate[None] = learning_rate_value

        stim_interval_slider = gui.slider_int("Stim Interval", stim_interval_value[None], 0, 50)
        stim_lattice_slider = gui.slider_int("Stim Lattice Size", stim_lattice_size[None], 1, 20)
        stim_interval_value[None] = stim_interval_slider
        stim_lattice_size[None] = stim_lattice_slider

        random_stimulation_toggle = gui.checkbox("Random Stimulation", random_stimulation[None] == 1)
        random_stimulation[None] = 1 if random_stimulation_toggle else 0

        gui.text("Homeostatic Plasticity")
        target_activity_value = gui.slider_float("Target Activity", target_activity[None], 0.0, 1.0)
        threshold_lr_value = gui.slider_float("Threshold Learning Rate", threshold_learning_rate[None], 0.0, 0.01)
        target_activity[None] = target_activity_value
        threshold_learning_rate[None] = threshold_lr_value

    # Calculate remaining time to achieve desired FPS
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_duration:
        time.sleep(frame_duration - elapsed_time)

    window.show()
