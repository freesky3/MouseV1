import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.v1model.geometry import L4, L2_3
from src.v1model.default_config import Config
from src.v1model.experimental_data import ExperimentalData

def visualize_network(l4, l2_3, save_path='output/test_result/test_geometry.png'):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_axes([0.05, 0.1, 0.70, 0.8], projection='3d')

    # ==============================
    # plot L4 layer (Z = 0)
    # ==============================
    l4_coords = l4.coords
    l4_tunings = l4.tunings
    
    # L4 untuned neurons (U) -> gray dots
    idx_l4_u = (l4_tunings == 'U')
    ax.scatter(l4_coords[idx_l4_u, 0], l4_coords[idx_l4_u, 1], np.full(np.sum(idx_l4_u), l4.z_pos), 
               c='lightgray', marker='o', s=15, alpha=0.5, label='L4 Untuned (X)')

    # L4 tuned neurons (T) -> color by pref_dir
    idx_l4_t = (l4_tunings == 'T')
    l4_pref_dirs = l4.pref_dirs[idx_l4_t]
    
    sc_l4 = ax.scatter(l4_coords[idx_l4_t, 0], l4_coords[idx_l4_t, 1], np.full(np.sum(idx_l4_t), l4.z_pos), 
                       c=l4_pref_dirs, cmap='hsv', marker='o', s=25, alpha=0.9, label='L4 Tuned (X)')

    # ==============================
    # plot L2/3 layer (Z = 0.5)
    # ==============================
    l23_coords = l2_3.coords
    l23_types = l2_3.types

    # L2/3 inhibitory neurons (I) -> blue triangles
    idx_l23_i = (l23_types == 'I')
    ax.scatter(l23_coords[idx_l23_i, 0], l23_coords[idx_l23_i, 1], np.full(np.sum(idx_l23_i), l2_3.z_pos), 
               c='blue', marker='v', s=20, alpha=0.7, label='L2/3 Inhibitory (I)')

    # L2/3 excitatory neurons (E) -> red squares
    idx_l23_e = (l23_types == 'E')
    ax.scatter(l23_coords[idx_l23_e, 0], l23_coords[idx_l23_e, 1], np.full(np.sum(idx_l23_e), l2_3.z_pos), 
               c='red', marker='s', s=20, alpha=0.7, label='L2/3 Excitatory (E)')

    # ==============================
    # chart settings and ratio adjustment
    # ==============================
    ax.set_title("3D Visualization of L4 and L2/3 Layers", pad=20)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position (Layer)")
    ax.set_zticks([l4.z_pos, l2_3.z_pos])
    ax.set_zticklabels(['L4', 'L2/3'])
    
    # force set the axis range to prevent the automatic scaling from causing the image to become narrow
    half_size = l4.region_size / 2.0
    ax.set_xlim(-half_size, half_size)
    ax.set_ylim(-half_size, half_size)
    ax.set_zlim(l4.z_pos - 0.1, l2_3.z_pos + 0.1)
    
    # force set the 3D box ratio
    ax.set_box_aspect((1, 1, 1.2)) 
    
    # make the 3D background panel transparent and the grid lines lighter
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.8)})
    ax.yaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.8)})
    ax.zaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.8)})

    # ==============================
    # integrate the right panel (legend + horizontal color bar)
    # ==============================
    # put the legend in the upper right position of the 3D figure, and add a title
    legend = ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.55), 
                       title="Neuron Properties", frameon=True)
    legend.get_title().set_fontweight('bold')
    
    # manually draw a small area below the legend to draw a horizontal color bar
    # parameters: [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.76, 0.45, 0.15, 0.02]) 
    cbar = fig.colorbar(sc_l4, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Preferred Direction (Radians)", fontsize=10)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    cfg = Config()
    exp_data = ExperimentalData(cfg)
    l4 = L4(cfg, exp_data)
    l2_3 = L2_3(cfg, exp_data)
    print(f"L4 neurons: {l4.n_side} * {l4.n_side}, L2/3 neurons: {l2_3.n_side} * {l2_3.n_side}")
    print(f"Ratio of L2/3 over L4: {l2_3.N / l4.N}")
    print(f"L4 tuned neurons: {np.sum(l4.tunings == 'T')}, {np.sum(l4.tunings == 'T')/l4.N*100}%")
    print(f"L4 untuned neurons: {np.sum(l4.tunings == 'U')}, {np.sum(l4.tunings == 'U')/l4.N*100}%")
    print(f"L2/3 excitatory neurons: {np.sum(l2_3.types == 'E')}, {np.sum(l2_3.types == 'E')/l2_3.N*100}%")
    print(f"L2/3 inhibitory neurons: {np.sum(l2_3.types == 'I')}, {np.sum(l2_3.types == 'I')/l2_3.N*100}%")
    visualize_network(l4, l2_3)

if __name__ == "__main__":
    main()