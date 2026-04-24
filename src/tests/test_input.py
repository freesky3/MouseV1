import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from src.v1model.default_config import Config
from src.v1model.experimental_data import ExperimentalData
from src.v1model.geometry import L4
from src.v1model.input import L4VisualInput

cfg = Config()

def plot_input(visual_input, theta_stim = np.pi/4, t_start = cfg.t_start, save_path='output/test_result/test_input_input.gif'):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(rf"Drifting Grating Stimulus $L(x,y,t)$, $\theta = {np.degrees(theta_stim):.1f}°$, $t = {t_start}$")
    ax.axis("off")

    # get initial frame and set imshow
    L_initial = visual_input.get_drifting_grating(theta_stim=theta_stim, t=t_start)

    # use extent to map pixel coordinates to actual physical coordinates [-fov_size/2, fov_size/2]
    fov = visual_input.fov_size
    img = ax.imshow(L_initial, cmap='gray', vmin=0, vmax=2, 
                    extent=[-fov/2, fov/2, -fov/2, fov/2])

    # animation update function
    def update(frame):
        # compute current time t based on frame number (can adjust time step according to omega)
        input_t = t_start + frame * 0.05 
        L_t = visual_input.get_drifting_grating(theta_stim=theta_stim, t=input_t)
        img.set_data(L_t)
        return [img]

    # create animation: 50 frames, 50 ms interval
    anim = FuncAnimation(fig, update, frames=50, interval=50, blit=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path)
    plt.close(fig)

def plot_gabor_rf_overlay(visual_input, theta_stim = np.pi/4, t_start = cfg.t_start, num_samples=40, save_path='output/test_result/test_input_gabor.png'):
    """
    overlay Gabor receptive fields' position, size and preferred orientation on the background grating.
    
    Args:
        theta_stim: background grating's stimulus orientation.
        num_samples: number of neurons to plot. Since there may be many neurons (e.g., 400),
                        drawing all of them will overlap, so randomly select some for visualization.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Gabor Receptive Fields Overlay (Stim: {np.degrees(theta_stim):.1f}°)")

    # 1. get and plot background grating at t=0
    L_bg = visual_input.get_drifting_grating(theta_stim=theta_stim, t=t_start)
    fov = visual_input.fov_size
    
    # use origin='lower' to ensure y-axis direction is consistent with actual coordinate system
    ax.imshow(L_bg, cmap='gray', vmin=0, vmax=visual_input.L0 * (1 + visual_input.epsilon),
                extent=[-fov/2, fov/2, -fov/2, fov/2], origin='lower', alpha=0.6)

    # 2. randomly sample neurons to prevent the screen from being too crowded
    if num_samples is None or num_samples >= visual_input.N:
        indices = np.arange(visual_input.N)
    else:
        indices = np.random.choice(visual_input.N, num_samples, replace=False)

    # 3. plot Gabor receptive fields
    for i in indices:
        x, y = visual_input.x_i[i], visual_input.y_i[i]
        is_tuned = visual_input.is_tuned[i]
        theta = visual_input.theta_i[i]

        # Gaussian envelope: exp(-(x'^2 + gamma * y'^2) / (2 * sigma^2))
        # 1-sigma boundary corresponds to the width and height of the ellipse
        width = 2 * visual_input.sigma
        height = 2 * visual_input.sigma / np.sqrt(visual_input.gamma) if visual_input.gamma > 0 else 2 * visual_input.sigma
        
        angle_deg = np.degrees(theta)
        color = 'crimson' if is_tuned else 'dodgerblue'

        # add receptive field envelope boundary (ellipse)
        ellipse = patches.Ellipse((x, y), width, height, angle=angle_deg, 
                                    edgecolor=color, facecolor='none', lw=2, alpha=0.8)
        ax.add_patch(ellipse)

        if is_tuned:
            # draw a line segment to indicate the preferred stripe orientation (perpendicular to k vector)
            # stripe orientation is theta + 90 degrees
            stripe_angle = theta + np.pi / 2
            dx = (height / 2) * np.cos(stripe_angle)
            dy = (height / 2) * np.sin(stripe_angle)
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=1.5, alpha=0.9)
        else:
            # non-tuned neurons are marked with a cross
            ax.plot(x, y, marker='+', color=color, markersize=8)

    ax.set_xlim(-fov/2, fov/2)
    ax.set_ylim(-fov/2, fov/2)
    ax.set_xlabel("X Position (Cortical Space)")
    ax.set_ylabel("Y Position (Cortical Space)")
    
    # add legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='crimson', lw=2, label='Tuned (T) RF'),
        Line2D([0], [0], marker='+', color='dodgerblue', lw=0, markersize=8, label='Untuned (U) RF')
    ]
    ax.legend(handles=custom_lines, loc='upper right', framealpha=0.9)

    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_response(visual_input, theta_stim = np.pi/4, t_start = cfg.t_start, save_path='output/test_result/test_input_response.png'):
    # get t=0 response of the network to the specified orientation grating
    r_X = visual_input.get_input_at_theta(theta_stim=theta_stim, t=t_start)

    plt.figure(figsize=(8, 6))

    # use scatter plot, c parameter is firing rate, cmap is a heatmap color
    scatter = plt.scatter(visual_input.x_i, visual_input.y_i, 
                        c=r_X, cmap='magma', s=80, 
                        edgecolors='gray', linewidth=0.5)

    # add color bar and label
    cbar = plt.colorbar(scatter)
    cbar.set_label('Firing Rate (Hz)', rotation=270, labelpad=15)

    plt.title(f"L4 Population Response (Stim: {np.degrees(theta_stim):.1f}°, $t = {t_start}$)")
    plt.xlabel("X Position (Cortical Space)")
    plt.ylabel("Y Position (Cortical Space)")

    # keep XY axis ratio consistent with the actual spatial mapping
    plt.axis('equal') 
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_phase_invariant_response(visual_input, theta_stim = np.pi/4, t_start = cfg.t_start, time_steps = 100, save_path='output/test_result/test_input_response_periodic.png'):
    """
    plot the phase-invariant response required for the steady-state network.
    calculate and display the maximum firing rate distribution over a full cycle of the drifting grating for the given stimulus orientation.
    """
    # 1. calculate the maximum firing response within one cycle
    cycle_duration = 2 * np.pi / visual_input.omega
    t_samples = np.linspace(0, cycle_duration, time_steps, endpoint=False)
    
    # iterate over each time step in the cycle to calculate the firing rate
    rates = [visual_input.get_input_at_theta(theta_stim, t_start + t) for t in t_samples]
    
    # take the maximum value along the time axis (axis=0) to get the phase-invariant response of each neuron, shape: (N_X,)
    r_X_max = np.max(rates, axis=0) 

    # 2. visualize the maximum response
    plt.figure(figsize=(8, 6))

    # use scatter plot, c parameter is the maximum firing rate r_X_max
    scatter = plt.scatter(visual_input.x_i, visual_input.y_i, 
                            c=r_X_max, cmap='magma', s=80, 
                            edgecolors='gray', linewidth=0.5)

    # add color bar and label
    cbar = plt.colorbar(scatter)
    cbar.set_label('Max Firing Rate (Hz)', rotation=270, labelpad=15)

    # update the title to reflect that this is the maximum response within the cycle
    plt.title(f"L4 Phase-Invariant Max Response (Stim: {np.degrees(theta_stim):.1f}°)")
    plt.xlabel("X Position (Cortical Space)")
    plt.ylabel("Y Position (Cortical Space)")

    # keep XY axis ratio consistent with the actual spatial mapping
    plt.axis('equal') 
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_all_neurons_firing_rate_over_cycle(visual_input, theta_stim = np.pi/4, t_start = cfg.t_start, time_steps = 100, save_path='output/test_result/test_input_rate_periodic.png'):
    """
    plot the firing rate of all neurons in L4 over a full cycle of a drifting grating.
    """
    # 1. calculate the time length of a full cycle
    cycle_duration = 2 * np.pi / visual_input.omega
    # generate time sampling points
    t_samples = np.linspace(0, cycle_duration, time_steps, endpoint=False)
    
    # 2. calculate the firing rate at each time step
    # rates will have shape (time_steps, N)
    rates = np.array([visual_input.get_input_at_theta(theta_stim, t_start + t) for t in t_samples])
    num_samples = 20
    sample_indices = np.random.choice(visual_input.N, num_samples, replace=False)
    
    # 3. visualize
    plt.figure(figsize=(10, 6))
    
    # plot the firing rate curves of all neurons
    # because there may be hundreds of neurons, use a low alpha value to prevent the screen from being blurred
    plt.plot(t_samples, rates[:, sample_indices], color='steelblue', alpha=0.5)
    
    # calculate and plot the population mean firing rate (optional, use a thick red line to represent)
    mean_rates = np.mean(rates, axis=1)
    plt.plot(t_samples, mean_rates, color='crimson', linewidth=2.5, label='Population Mean')

    # decorate the plot
    plt.title(f"Firing Rates of All L4 Neurons Over One Cycle (Stim: {np.degrees(theta_stim):.1f}°, $t = {t_start}$)")
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (Hz)")
    plt.xlim(0, cycle_duration)
    
    # avoid duplicate legends (because plt.plot drawing a matrix will produce N labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    # only keep the legend of the mean line, and manually add a legend representing individual neurons
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='steelblue', alpha=0.5, lw=1.5),
                    handles[-1]] # handles[-1] is the mean line
    plt.legend(custom_lines, ['Individual Neurons', 'Population Mean'], loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    cfg = Config()
    exp_data = ExperimentalData(cfg)
    l4_layer = L4(cfg, exp_data)
    visual_input = L4VisualInput(l4_layer, cfg)
    plot_input(visual_input)
    plot_gabor_rf_overlay(visual_input)
    plot_response(visual_input)
    plot_phase_invariant_response(visual_input)
    plot_all_neurons_firing_rate_over_cycle(visual_input)


if __name__ == "__main__":
    main()