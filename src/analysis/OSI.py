import numpy as np

def get_osi(responses, theta_angles, threshold=0.2):
    """
    use global-gOSI
    responses: shape (N_neurons, N_orientations)
    theta_angles: shape (N_orientations,)
    """
    complex_vectors = responses * np.exp(2j * theta_angles)
    vector_sum = np.sum(complex_vectors, axis=1)
    vector_sum_abs = np.abs(vector_sum)
    pref_ori = np.angle(vector_sum) / 2
    scalar_sum = np.sum(responses, axis=1) + 1e-8
    
    osi = vector_sum_abs / scalar_sum
    pref_ori[osi < threshold] = np.nan
    
    return osi, pref_ori

def plot_osi_results(osi, pref_ori, neuron_coords, save_dir="output/analysis"):
    import matplotlib.pyplot as plt
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # ---- 1. Histogram of OSI ----
    plt.figure(figsize=(6, 4))
    plt.hist(osi, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("OSI", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Global OSI", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "osi_histogram.png"), dpi=300)
    plt.close()
    
    # ---- 2. Spatial Distribution of OSI ----
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(neuron_coords[:, 0], neuron_coords[:, 1], c=osi, cmap='viridis', s=20)
    plt.colorbar(sc, label='OSI')
    plt.title("Spatial Distribution of OSI", fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "osi_spatial_distribution.png"), dpi=300)
    plt.close()
    
    # ---- 3. Spatial Distribution of Preferred Orientation ----
    plt.figure(figsize=(6, 5))
    # valid orientation cells
    valid = ~np.isnan(pref_ori)
    
    # draw low OSI cells in background
    plt.scatter(neuron_coords[~valid, 0], neuron_coords[~valid, 1], 
                color='lightgray', s=10, alpha=0.5, label='Low OSI')
    
    # draw cells with valid preferred orientation
    if np.sum(valid) > 0:
        sc = plt.scatter(neuron_coords[valid, 0], neuron_coords[valid, 1], 
                         c=pref_ori[valid], cmap='hsv', s=20)
        plt.colorbar(sc, label='Preferred Orientation (rad)')
        
    plt.title("Preferred Orientation", fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pref_ori_spatial_distribution.png"), dpi=300)
    plt.close()

    # ---- 4. Centers of Preferred Orientations ----
    plt.figure(figsize=(5, 5))
    
    # 对于低OSI细胞，绘制和参考图一样的灰色空心圆
    plt.scatter(neuron_coords[~valid, 0], neuron_coords[~valid, 1], 
                facecolors='none', edgecolors='lightgray', s=30, alpha=0.8)
    
    if np.sum(valid) > 0:
        # 将连续角度映射到 [0, pi) 范围内，并分为4个核心角度
        pref_ori_mod = np.mod(pref_ori[valid], np.pi)
        n_bins = 4
        bins = np.linspace(0, np.pi, n_bins + 1)
        # 用粉、蓝、紫、黄四个颜色，以还原图中的风格
        colors = ['#e377c2', '#1f77b4', '#8A2BE2', '#DAA520']
        
        valid_coords = neuron_coords[valid]
        for i in range(n_bins):
            # 获取属于当前角度区间的细胞
            if i == n_bins - 1:
                idx = (pref_ori_mod >= bins[i]) & (pref_ori_mod <= bins[i+1])
            else:
                idx = (pref_ori_mod >= bins[i]) & (pref_ori_mod < bins[i+1])
                
            group_coords = valid_coords[idx]
            if len(group_coords) > 0:
                # 画对应角度的细胞
                plt.scatter(group_coords[:, 0], group_coords[:, 1], 
                            color=colors[i], s=50, alpha=0.9, edgecolors='none')
                
                # 计算属于该角度类的细胞的平均位置（重心），绘制大叉
                mean_pos = np.mean(group_coords, axis=0)
                plt.scatter(mean_pos[0], mean_pos[1], 
                            color=colors[i], marker='x', s=400, linewidths=4, zorder=10)

    # 去掉边框和坐标轴，和参考图风格保持一致
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ori_centers_spatial.png"), dpi=300, transparent=False)
    plt.close()
