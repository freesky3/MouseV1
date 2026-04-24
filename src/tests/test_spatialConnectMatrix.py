from src.v1model.default_config import Config
from src.v1model.experimental_data import ExperimentalData
from src.v1model.geometry import L4, L2_3
from src.v1model.SpatialConnectMatrix import SpatialConnectMatrix
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

from mpl_toolkits.mplot3d.art3d import Line3DCollection
def plot_weights(scm, save_path="output/test_result/test_scm_weights.png"):
    active_weights = scm.QJ_ij[scm.Q == 1]
    
    if len(active_weights) > 0:
        # 【优化1】分位数调低到 95% 或 90%，让大部分连接点的颜色更深、更红/更蓝
        vmax = np.percentile(np.abs(active_weights), 95)
        if vmax == 0: vmax = 1.0
    else:
        vmax = 1.0

    # 【优化2】调大画布并大幅提高 dpi=300，防止像素点被抗锯齿糊掉
    plt.figure(figsize=(12, 10)) 
    
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # 【优化3】使用 'nearest' 替代 'none'，在密集矩阵缩放时表现更好
    im = plt.imshow(scm.QJ_ij, cmap='coolwarm', norm=norm, aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='Synaptic Weight ($J_{ij}$)')

    plt.title("Synaptic Weight Matrix ($QJ_{ij}$)", fontsize=14)
    plt.xlabel("Target Neuron Index (L2/3 + L4)", fontsize=12)
    plt.ylabel("Source Neuron Index (L2/3)", fontsize=12)

    N_L23 = scm.l23.N
    plt.axvline(x=N_L23, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=N_L23, color='black', linestyle='--', linewidth=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()



def visualize_connectivity(l4, l23, scm, selected_idx, layer='L23', direction='in', save_path="output/test_result/test_scm_connectivity"):
    """
    可视化选定神经元的突触连接结构。
    
    参数:
    - selected_idx: 在其所在层内的局部索引。
    - layer: 'L23' 或 'L4'。
    - direction: 'in' (传入连接) 或 'out' (传出连接)。
    """
    save_path = save_path + f'_{layer}_{direction}.png'
    # 调整画布比例，使其更宽，便于右侧放置图例和颜色条
    fig = plt.figure(figsize=(12, 7))
    
    # 扩大 3D 图的显示范围，减少四周的留白
    # [left, bottom, width, height]
    ax = fig.add_axes([-0.05, 0.05, 0.85, 0.95], projection='3d')

    N_L23 = l23.N
    N_L4 = l4.N
    
    l23_coords = l23.coords
    l4_coords = l4.coords
    all_coords = np.vstack([l23_coords, l4_coords])
    
    # 绘制背景层 (淡化未连接的神经元)
    ax.scatter(l4_coords[:, 0], l4_coords[:, 1], l4.z_pos, 
               c='lightgray', marker='o', s=10, alpha=0.1)
    ax.scatter(l23_coords[:, 0], l23_coords[:, 1], l23.z_pos, 
               c='gray', marker='^', s=10, alpha=0.1)
               
    cmap = cm.coolwarm

    if direction == 'in':
        if layer != 'L23':
            raise ValueError("此模型中只有 L2/3 神经元接收传入连接。")
            
        target_coord = l23_coords[selected_idx]
        ax.scatter(target_coord[0], target_coord[1], l23.z_pos, c='gold', marker='*', s=400, edgecolor='k', zorder=5)
        
        connected_idx = np.where(scm.Q[selected_idx, :] == 1)[0]
        sources = all_coords[connected_idx]
        conn_weights = scm.QJ_ij[selected_idx, connected_idx]
        
        vmax = np.max(np.abs(conn_weights)) if len(conn_weights) > 0 else 1
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        for src, w, idx in zip(sources, conn_weights, connected_idx):
            color = cmap(norm(w))
            src_z = l23.z_pos if idx < N_L23 else l4.z_pos
            ax.plot([src[0], target_coord[0]], 
                    [src[1], target_coord[1]], 
                    [src_z, l23.z_pos], 
                    color=color, alpha=0.6, linewidth=1.5)
            # 标记源神经元
            ax.scatter(src[0], src[1], src_z, color=color, marker='o' if src_z == l4.z_pos else '^', 
                       s=40, edgecolor='k')
            
    elif direction == 'out':
        if layer == 'L23':
            source_global_idx = selected_idx
            source_coord = l23_coords[selected_idx]
            source_z = l23.z_pos
        elif layer == 'L4':
            source_global_idx = N_L23 + selected_idx
            source_coord = l4_coords[selected_idx]
            source_z = l4.z_pos
            
        ax.scatter(*source_coord, c='gold', marker='*', s=400, edgecolor='k', zorder=5)
            
        connected_idx = np.where(scm.Q[:, source_global_idx] == 1)[0]
        targets = l23_coords[connected_idx]
        conn_weights = scm.QJ_ij[connected_idx, source_global_idx]
        
        vmax = np.max(np.abs(conn_weights)) if len(conn_weights) > 0 else 1
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        for tgt, w in zip(targets, conn_weights):
            color = cmap(norm(w))
            ax.plot([source_coord[0], tgt[0]], 
                    [source_coord[1], tgt[1]], 
                    [source_z, l23.z_pos], 
                    color=color, alpha=0.6, linewidth=1.5)
            ax.scatter(tgt[0], tgt[1], l23.z_pos, color=color, marker='^', s=40, edgecolor='k')

    # 图表装饰
    ax.set_title(f"Synaptic Connections ({direction.upper()}) for Neuron {selected_idx} in {layer}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_zticks([l4.z_pos, l23.z_pos])
    ax.set_zticklabels(['L4', 'L2/3'])
    ax.set_box_aspect((1, 1, 0.4)) 
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.8)})
    ax.yaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.8)})
    ax.zaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.8)})

    # ==========================================
    # 1. 添加形状图例 (在图表右上方)
    # ==========================================
    star_label = 'Target' if direction == 'in' else 'Source'
    legend_elements = [
        mlines.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='k', markersize=15, label=f'{star_label} Neuron'),
        mlines.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=9, label='L2/3 Neuron'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=9, label='L4 Neuron')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 0.95), title="Marker Shapes", frameon=True)

    # ==========================================
    # 2. 添加颜色条 (在图例下方)
    # ==========================================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.83, 0.25, 0.02, 0.45])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Synaptic Weight ($J_{ij}$)", fontsize=11)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()



def interactive_connectivity(l4, l23, scm, direction='in'):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([-0.05, 0.05, 0.85, 0.95], projection='3d')
    
    if hasattr(ax, 'disable_mouse_rotation'):
        ax.disable_mouse_rotation() 
    ax.set_navigate(False)

    N_L23 = l23.N
    N_L4 = l4.N
    l23_coords = l23.coords
    l4_coords = l4.coords
    all_coords = np.vstack([l23_coords, l4_coords])
    
    # 修改背景点的 Z 坐标
    ax.scatter(l4_coords[:, 0], l4_coords[:, 1], l4.z_pos, 
               c='gray', marker='o', s=15, alpha=0.08, zorder=1)
    ax.scatter(l23_coords[:, 0], l23_coords[:, 1], l23.z_pos, 
               c='gray', marker='^', s=15, alpha=0.08, zorder=1)

    valid_mask = scm.Q == 1
    active_weights = scm.QJ_ij[valid_mask]
    if len(active_weights) > 0:
        vmax = np.percentile(np.abs(active_weights), 98)
        if vmax == 0: vmax = 1.0
    else:
        vmax = 1.0
        
    cmap = cm.coolwarm
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    lines_collection = None
    target_scatter = None
    connected_scatters = [] 
    current_hover = None

    ax.set_title(f"Interactive Synaptic Connections ({direction.upper()})\n[Hover over any neuron to reveal its synapses]", pad=20)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_zticks([l4.z_pos, l23.z_pos])
    ax.set_zticklabels(['L4', 'L2/3'])
    ax.set_box_aspect((1, 1, 0.4)) 
    
    legend_elements = [
        mlines.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='k', markersize=15, label='Hovered Neuron'),
        mlines.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=9, label='L2/3 Neuron'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=9, label='L4 Neuron')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 0.95), frameon=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.83, 0.25, 0.02, 0.45])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Synaptic Weight ($J_{ij}$)", fontsize=11)

    # 修正坐标映射函数，传入 z_pos
    def get_screen_pixels(coords_2d, z_pos):
        z_arr = np.full(len(coords_2d), z_pos)
        try:
            x2, y2, _ = proj3d.proj_transform(coords_2d[:,0], coords_2d[:,1], z_arr, ax.get_proj())
        except TypeError:
            x2, y2 = np.zeros(len(coords_2d)), np.zeros(len(coords_2d))
            for i in range(len(coords_2d)):
                x2[i], y2[i], _ = proj3d.proj_transform(coords_2d[i,0], coords_2d[i,1], z_arr[i], ax.get_proj())
        return ax.transData.transform(np.column_stack([x2, y2]))

    def clear_highlights():
        nonlocal lines_collection, target_scatter
        if lines_collection is not None:
            try: lines_collection.remove()
            except: pass
            lines_collection = None
        if target_scatter is not None:
            try: target_scatter.remove()
            except: pass
            target_scatter = None
        for sc in connected_scatters:
            try: sc.remove()
            except: pass
        connected_scatters.clear()

    def draw_connections(selected_idx, layer):
        nonlocal lines_collection, target_scatter
        clear_highlights()
        
        lines_coords = []
        colors = []
        
        if layer == 'L2/3':
            coord = l23_coords[selected_idx]
            global_idx = selected_idx
            coord_z = l23.z_pos
        else:
            coord = l4_coords[selected_idx]
            global_idx = N_L23 + selected_idx
            coord_z = l4.z_pos
            
        target_scatter = ax.scatter(coord[0], coord[1], coord_z, c='gold', marker='*', s=450, edgecolor='k', linewidths=1.5, zorder=100)
        coord_3d = [coord[0], coord[1], coord_z]

        connected_pts_l23 = []
        connected_pts_l4 = []
        connected_colors_l23 = []
        connected_colors_l4 = []

        if direction == 'in':
            if layer == 'L2/3': 
                connected_idx = np.where(scm.Q[selected_idx, :] == 1)[0]
                sources = all_coords[connected_idx]
                conn_weights = scm.QJ_ij[selected_idx, connected_idx]
                for src, idx, w in zip(sources, connected_idx, conn_weights):
                    src_z = l23.z_pos if idx < N_L23 else l4.z_pos
                    src_3d = [src[0], src[1], src_z]
                    lines_coords.append([src_3d, coord_3d]) 
                    col = cmap(norm(w))
                    colors.append(col)
                    if idx < N_L23:
                        connected_pts_l23.append(src_3d)
                        connected_colors_l23.append(col)
                    else:
                        connected_pts_l4.append(src_3d)
                        connected_colors_l4.append(col)
        
        elif direction == 'out':
            connected_idx = np.where(scm.Q[:, global_idx] == 1)[0]
            targets = l23_coords[connected_idx]
            conn_weights = scm.QJ_ij[connected_idx, global_idx]
            for tgt, idx, w in zip(targets, connected_idx, conn_weights):
                tgt_3d = [tgt[0], tgt[1], l23.z_pos]
                lines_coords.append([coord_3d, tgt_3d])
                col = cmap(norm(w))
                colors.append(col)
                connected_pts_l23.append(tgt_3d)
                connected_colors_l23.append(col)

        if connected_pts_l23:
            pts = np.array(connected_pts_l23)
            sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=connected_colors_l23, marker='^', s=45, edgecolor='k', zorder=50)
            connected_scatters.append(sc)
        if connected_pts_l4:
            pts = np.array(connected_pts_l4)
            sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=connected_colors_l4, marker='o', s=45, edgecolor='k', zorder=50)
            connected_scatters.append(sc)

        if lines_coords:
            lines_collection = Line3DCollection(lines_coords, colors=colors, alpha=0.9, linewidths=2.0, zorder=10)
            ax.add_collection3d(lines_collection)
            
        fig.canvas.draw_idle()

    def on_hover(event):
        nonlocal current_hover
        
        if event.inaxes != ax or event.x is None or event.y is None:
            return
            
        mouse_pixel = np.array([event.x, event.y])
        l23_pixels = get_screen_pixels(l23_coords, l23.z_pos)
        l4_pixels = get_screen_pixels(l4_coords, l4.z_pos)
        
        dist_l23 = np.linalg.norm(l23_pixels - mouse_pixel, axis=1)
        dist_l4 = np.linalg.norm(l4_pixels - mouse_pixel, axis=1)
        
        min_l23 = np.min(dist_l23)
        min_l4 = np.min(dist_l4)
        
        hit_idx = None
        layer = None
        
        THRESHOLD = 15
        
        if min_l23 < THRESHOLD and min_l23 <= min_l4:
            hit_idx = np.argmin(dist_l23)
            layer = 'L2/3'
        elif min_l4 < THRESHOLD and min_l4 < min_l23:
            hit_idx = np.argmin(dist_l4)
            layer = 'L4'
            
        if hit_idx is not None:
            new_state = (hit_idx, layer)
            if new_state != current_hover:
                current_hover = new_state
                draw_connections(hit_idx, layer)
        else:
            if current_hover is not None:
                current_hover = None
                clear_highlights()
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    plt.show()


def main():
    cfg = Config()
    exp_data = ExperimentalData(cfg)
    l4 = L4(cfg, exp_data)
    l23 = L2_3(cfg, exp_data)
    scm = SpatialConnectMatrix(l23, l4, cfg, exp_data)
    
    # ==========================================
    # 打印各类神经元的实际连接概率 (Empirical Probabilities)
    # ==========================================
    p_EE_actual = np.mean(scm.Q[np.ix_(scm.idx_E, scm.idx_E)])
    p_EI_actual = np.mean(scm.Q[np.ix_(scm.idx_E, scm.idx_I)])
    p_EX_actual = np.mean(scm.Q[np.ix_(scm.idx_E, scm.idx_X)])
    
    p_IE_actual = np.mean(scm.Q[np.ix_(scm.idx_I, scm.idx_E)])
    p_II_actual = np.mean(scm.Q[np.ix_(scm.idx_I, scm.idx_I)])
    p_IX_actual = np.mean(scm.Q[np.ix_(scm.idx_I, scm.idx_X)])

    print("\n" + "="*45)
    print("实际生成的突触连接概率 vs 目标概率 (Target):")
    print(f"p_EE : {p_EE_actual:.4f}  (Target: {scm.data.p_EE:.4f})")
    print(f"p_EI : {p_EI_actual:.4f}  (Target: {scm.data.p_EI:.4f})")
    print(f"p_EX : {p_EX_actual:.4f}  (Target: {scm.data.p_EX:.4f})")
    print(f"p_IE : {p_IE_actual:.4f}  (Target: {scm.data.p_IE:.4f})")
    print(f"p_II : {p_II_actual:.4f}  (Target: {scm.data.p_II:.4f})")
    print(f"p_IX : {p_IX_actual:.4f}  (Target: {scm.data.p_IX:.4f})")
    print("="*45 + "\n")
    
    # 设置打印选项
    np.set_printoptions(
        precision=2,       # 小数点后只保留 2 位
        suppress=True,     # 禁止使用科学计数法 (例如 1e-4 会变成 0.00)
        linewidth=150,     # 单行最多容纳的字符数，调大可以防止频繁换行
        edgeitems=5        # 缩略号 (...) 前后显示的行/列数，默认是 3
    )

    print("=== 连接矩阵 Q ===")
    print(scm.Q)
    print("\n=== 权重矩阵 QJ_ij ===")
    print(scm.QJ_ij)

    plot_weights(scm)
    visualize_connectivity(l4, l23, scm, selected_idx=150, layer='L4', direction='out')
    visualize_connectivity(l4, l23, scm, selected_idx=200, layer='L23', direction='in')
    interactive_connectivity(l4, l23, scm, direction='in')   # 查看各个神经元的感受野 (接收)
    interactive_connectivity(l4, l23, scm, direction='out')   # 查看各个神经元的感受野 (接收)

if __name__ == "__main__":
    main()
