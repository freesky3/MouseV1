from experimental_data import ExperimentalData
from geometry import L2_3, L4
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

from mpl_toolkits.mplot3d.art3d import Line3DCollection

class SpatialConnectMatrix:
    def __init__(self, l23_layer, l4_layer, experimental_data, J, g, sigma_narrow, sigma_broad, kappa):
        self.l23 = l23_layer
        self.l4 = l4_layer
        self.data = experimental_data

        N_L23 = self.l23.N
        N_L4 = self.l4.N
        idx_L23 = np.arange(N_L23)
        idx_L4 = np.arange(N_L23, N_L23 + N_L4)

        is_E = np.array([n.n_type == 'E' for n in self.l23.neurons])
        is_I = np.array([n.n_type == 'I' for n in self.l23.neurons])
        self.idx_E = idx_L23[is_E]
        self.idx_I = idx_L23[is_I]
        self.idx_X = idx_L4
        
        self.sigma_narrow = 0.09  # 例如 50 um 对应 0.05 region_size
        self.sigma_broad = 0.25   # 例如 200 um 对应 0.20 region_size
        self.kappa = 0.8          # 窄连接占据的主导比例

        self.J = J
        self.g = g

        self.Q = self.sample_matrix()
        self.J_ij = self.sample_J()
        self.QJ_ij = self.Q * self.J_ij
        
    def spatial_kernel(self, distance_matrix):
        dist_sq = distance_matrix ** 2
        narrow_gauss = np.exp(-dist_sq / (2 * self.sigma_narrow**2))
        broad_gauss = np.exp(-dist_sq / (2 * self.sigma_broad**2))
        return self.kappa * narrow_gauss + (1 - self.kappa) * broad_gauss
    
    def _normalize_probabilities(self, S_matrix, target_p):
        """
        使用二分查找寻找最佳缩放系数，
        确保截断后的概率矩阵平均值严格等于 target_p。
        """
        # 如果目标概率为 0，直接返回全 0 矩阵
        if target_p <= 0:
            return np.zeros_like(S_matrix)
            
        low, high = 0.0, 10000.0  # 缩放系数的搜索范围
        best_P = None
        
        for _ in range(30):  # 30次迭代精度已足够高
            mid = (low + high) / 2
            P_temp = np.clip(mid * S_matrix, 0.0, 1.0)
            current_mean = np.mean(P_temp)
            
            if current_mean > target_p:
                high = mid
            else:
                low = mid
            best_P = P_temp
            
        return best_P


    def sample_matrix(self):
        N_L23 = self.l23.N
        N_L4 = self.l4.N
        N_total = N_L23 + N_L4
        
        # 初始化连接矩阵 Q
        Q = np.full((N_L23, N_total), np.nan)
        
        # 1. 提取各类神经元的全局索引
        idx_L23 = np.arange(N_L23)
        idx_L4 = np.arange(N_L23, N_total) # L4 的索引接在 L2/3 之后
        
        idx_E = self.idx_E
        idx_I = self.idx_I
        idx_X = self.idx_X
        local_idx_X = np.arange(N_L4)


        dist_L23 = self.l23.get_distance_matrix(periodic=True)
        dist_L4_to_L23 = self.l23.get_distance_to(self.l4, periodic=True) # Shape: (N_L23, N_L4)

        d_EE = dist_L23[np.ix_(idx_E, idx_E)]
        d_EI = dist_L23[np.ix_(idx_E, idx_I)]
        d_EX = dist_L4_to_L23[np.ix_(idx_E, local_idx_X)]

        d_IE = dist_L23[np.ix_(idx_I, idx_E)]
        d_II = dist_L23[np.ix_(idx_I, idx_I)]
        d_IX = dist_L4_to_L23[np.ix_(idx_I, local_idx_X)]
        
        # calculate the spatial kernel for the given distance matrix
        S_EE = self.spatial_kernel(d_EE)
        S_EI = self.spatial_kernel(d_EI)
        S_EX = self.spatial_kernel(d_EX)
        S_IE = self.spatial_kernel(d_IE)
        S_II = self.spatial_kernel(d_II)
        S_IX = self.spatial_kernel(d_IX)

        # normalize the spatial kernel
        S_EE = S_EE / np.mean(S_EE)
        S_EI = S_EI / np.mean(S_EI)
        S_EX = S_EX / np.mean(S_EX)
        S_IE = S_IE / np.mean(S_IE)
        S_II = S_II / np.mean(S_II)
        S_IX = S_IX / np.mean(S_IX)

        np.fill_diagonal(S_EE, 0.0)
        np.fill_diagonal(S_II, 0.0)


        P_EE_matrix = self._normalize_probabilities(S_EE, self.data.p_EE)
        P_EI_matrix = self._normalize_probabilities(S_EI, self.data.p_EI)
        P_EX_matrix = self._normalize_probabilities(S_EX, self.data.p_EX)
        
        P_IE_matrix = self._normalize_probabilities(S_IE, self.data.p_IE)
        P_II_matrix = self._normalize_probabilities(S_II, self.data.p_II)
        P_IX_matrix = self._normalize_probabilities(S_IX, self.data.p_IX)

        
        
        # E. 采样
        Q[np.ix_(idx_E, idx_E)] = np.random.binomial(1, P_EE_matrix)
        Q[np.ix_(idx_E, idx_I)] = np.random.binomial(1, P_EI_matrix)
        Q[np.ix_(idx_E, idx_X)] = np.random.binomial(1, P_EX_matrix)
        Q[np.ix_(idx_I, idx_E)] = np.random.binomial(1, P_IE_matrix)
        Q[np.ix_(idx_I, idx_I)] = np.random.binomial(1, P_II_matrix)
        Q[np.ix_(idx_I, idx_X)] = np.random.binomial(1, P_IX_matrix)

        return Q

    def sample_J(self):
        #J=3.;g=1.5;
        idx_E = self.idx_E
        idx_I = self.idx_I
        idx_X = self.idx_X

        J_ij=np.full(self.Q.shape, np.nan)

        J_ij[np.ix_(idx_E,idx_E)] = self.J  * np.random.choice(self.data.sampled_J_EE, size=np.shape(self.Q[np.ix_(idx_E,idx_E)]))
        J_ij[np.ix_(idx_E,idx_I)] = -self.J*self.g*np.random.choice(self.data.sampled_J_EI, size=np.shape(self.Q[np.ix_(idx_E,idx_I)]))
        J_ij[np.ix_(idx_E,idx_X)] = self.J  * np.random.choice(self.data.sampled_J_EX, size=np.shape(self.Q[np.ix_(idx_E,idx_X)]))

        J_ij[np.ix_(idx_I,idx_E)] = self.J  * np.random.choice(self.data.sampled_J_IE, size=np.shape(self.Q[np.ix_(idx_I,idx_E)]))
        J_ij[np.ix_(idx_I,idx_I)] = -self.J*self.g*np.random.choice(self.data.sampled_J_II, size=np.shape(self.Q[np.ix_(idx_I,idx_I)]))
        J_ij[np.ix_(idx_I,idx_X)] = self.J  * np.random.choice(self.data.sampled_J_IX, size=np.shape(self.Q[np.ix_(idx_I,idx_X)]))

        return J_ij

    def plot_weights(self):
        active_weights = self.QJ_ij[self.Q == 1]
        
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
        im = plt.imshow(self.QJ_ij, cmap='coolwarm', norm=norm, aspect='auto', interpolation='nearest')
        plt.colorbar(im, label='Synaptic Weight ($J_{ij}$)')

        plt.title("Synaptic Weight Matrix ($QJ_{ij}$)", fontsize=14)
        plt.xlabel("Target Neuron Index (L2/3 + L4)", fontsize=12)
        plt.ylabel("Source Neuron Index (L2/3)", fontsize=12)

        N_L23 = self.l23.N
        plt.axvline(x=N_L23, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=N_L23, color='black', linestyle='--', linewidth=1)

        plt.show()



def visualize_connectivity(l4, l23, scm, selected_idx, layer='L2/3', direction='in'):
    """
    可视化选定神经元的突触连接结构。
    
    参数:
    - selected_idx: 在其所在层内的局部索引。
    - layer: 'L2/3' 或 'L4'。
    - direction: 'in' (传入连接) 或 'out' (传出连接)。
    """
    # 调整画布比例，使其更宽，便于右侧放置图例和颜色条
    fig = plt.figure(figsize=(12, 7))
    
    # 扩大 3D 图的显示范围，减少四周的留白
    # [left, bottom, width, height]
    ax = fig.add_axes([-0.05, 0.05, 0.85, 0.95], projection='3d')

    N_L23 = l23.N
    N_L4 = l4.N
    
    l23_coords = np.array([n.pos for n in l23.neurons])
    l4_coords = np.array([n.pos for n in l4.neurons])
    all_coords = np.vstack([l23_coords, l4_coords])
    
    # 绘制背景层 (淡化未连接的神经元)
    ax.scatter(l4_coords[:, 0], l4_coords[:, 1], l4_coords[:, 2], 
               c='lightgray', marker='o', s=10, alpha=0.1)
    ax.scatter(l23_coords[:, 0], l23_coords[:, 1], l23_coords[:, 2], 
               c='gray', marker='^', s=10, alpha=0.1)
               
    cmap = cm.coolwarm

    if direction == 'in':
        if layer != 'L2/3':
            raise ValueError("此模型中只有 L2/3 神经元接收传入连接。")
            
        target_coord = l23_coords[selected_idx]
        ax.scatter(*target_coord, c='gold', marker='*', s=400, edgecolor='k', zorder=5)
        
        connected_idx = np.where(scm.Q[selected_idx, :] == 1)[0]
        sources = all_coords[connected_idx]
        conn_weights = scm.QJ_ij[selected_idx, connected_idx]
        
        vmax = np.max(np.abs(conn_weights)) if len(conn_weights) > 0 else 1
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        for src, w in zip(sources, conn_weights):
            color = cmap(norm(w))
            ax.plot([src[0], target_coord[0]], 
                    [src[1], target_coord[1]], 
                    [src[2], target_coord[2]], 
                    color=color, alpha=0.6, linewidth=1.5)
            # 标记源神经元
            ax.scatter(*src, color=color, marker='o' if src[2] == l4.z_pos else '^', 
                       s=40, edgecolor='k')
            
    elif direction == 'out':
        if layer == 'L2/3':
            source_global_idx = selected_idx
            source_coord = l23_coords[selected_idx]
        elif layer == 'L4':
            source_global_idx = N_L23 + selected_idx
            source_coord = l4_coords[selected_idx]
            
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
                    [source_coord[2], tgt[2]], 
                    color=color, alpha=0.6, linewidth=1.5)
            ax.scatter(*tgt, color=color, marker='^', s=40, edgecolor='k')

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
    
    plt.show()



def interactive_connectivity(l4, l23, scm, direction='in'):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([-0.05, 0.05, 0.85, 0.95], projection='3d')
    
    # 锁定视角
    if hasattr(ax, 'disable_mouse_rotation'):
        ax.disable_mouse_rotation() 
    ax.set_navigate(False)

    N_L23 = l23.N
    N_L4 = l4.N
    l23_coords = np.array([n.pos for n in l23.neurons])
    l4_coords = np.array([n.pos for n in l4.neurons])
    all_coords = np.vstack([l23_coords, l4_coords])
    
    # 【修复1：大幅降低背景神经元透明度，防止遮挡底层】
    ax.scatter(l4_coords[:, 0], l4_coords[:, 1], l4_coords[:, 2], 
               c='gray', marker='o', s=15, alpha=0.08, zorder=1)
    ax.scatter(l23_coords[:, 0], l23_coords[:, 1], l23_coords[:, 2], 
               c='gray', marker='^', s=15, alpha=0.08, zorder=1)

    # 【修复2：精准计算存在的连接权重，排除极值干扰以显现颜色】
    valid_mask = scm.Q == 1
    active_weights = scm.QJ_ij[valid_mask]
    if len(active_weights) > 0:
        # 使用 98% 分位数防止离群值将正常权重压缩为白色
        vmax = np.percentile(np.abs(active_weights), 98)
        if vmax == 0: vmax = 1.0
    else:
        vmax = 1.0
        
    cmap = cm.coolwarm
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    lines_collection = None
    target_scatter = None
    connected_scatters = [] # 用于管理动态生成的高亮连接点
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

    def get_screen_pixels(coords_3d):
        """将 3D 坐标转换为当前的 2D 屏幕像素坐标"""
        try:
            x2, y2, _ = proj3d.proj_transform(coords_3d[:,0], coords_3d[:,1], coords_3d[:,2], ax.get_proj())
        except TypeError:
            x2, y2 = np.zeros(len(coords_3d)), np.zeros(len(coords_3d))
            for i in range(len(coords_3d)):
                x2[i], y2[i], _ = proj3d.proj_transform(coords_3d[i,0], coords_3d[i,1], coords_3d[i,2], ax.get_proj())
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
        else:
            coord = l4_coords[selected_idx]
            global_idx = N_L23 + selected_idx
            
        # 【修复1：高亮星星加粗放大，并置于最顶层】
        target_scatter = ax.scatter(*coord, c='gold', marker='*', s=450, edgecolor='k', linewidths=1.5, zorder=100)

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
                    lines_coords.append([src, coord]) 
                    col = cmap(norm(w))
                    colors.append(col)
                    if idx < N_L23:
                        connected_pts_l23.append(src)
                        connected_colors_l23.append(col)
                    else:
                        connected_pts_l4.append(src)
                        connected_colors_l4.append(col)
        
        elif direction == 'out':
            connected_idx = np.where(scm.Q[:, global_idx] == 1)[0]
            targets = l23_coords[connected_idx]
            conn_weights = scm.QJ_ij[connected_idx, global_idx]
            for tgt, idx, w in zip(targets, connected_idx, conn_weights):
                lines_coords.append([coord, tgt])
                col = cmap(norm(w))
                colors.append(col)
                connected_pts_l23.append(tgt)
                connected_colors_l23.append(col)

        # 【修复3：显式绘制连接的端点，加黑边以突显于背景之上】
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
        l23_pixels = get_screen_pixels(l23_coords)
        l4_pixels = get_screen_pixels(l4_coords)
        
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

if __name__ == "__main__":
    config = {
        "L4_n_side": 20,
        "L4_region_size": 1,
        "L2_3_region_size": 1,
        "L4_z_pos": 0,
        "L2_3_z_pos": 0.1,
        "pT_X": 0.5714,
        "pU_X": 0.4286,
        "Theta": np.linspace(0, np.pi, 16, endpoint=False), 
        "pE": 0.8475,
        "pI": 0.1525,
        "p_EE": 0.1,

        "J": 3,
        "g": 1.5,
        "sigma_narrow": 0.09,
        "sigma_broad": 0.25,
        "kappa": 0.8,
    }
    exp_data = ExperimentalData(config["L4_n_side"]**2, config["p_EE"])
    l4 = L4(n_side=config["L4_n_side"], region_size=config["L4_region_size"], z_pos=config["L4_z_pos"], pT_X=config["pT_X"], pU_X=config["pU_X"], Theta=config["Theta"])
    l23 = L2_3(n_side=exp_data.L2_3_n_side, region_size=config["L2_3_region_size"], z_pos=config["L2_3_z_pos"], pE=config["pE"], pI=config["pI"])
    scm = SpatialConnectMatrix(l23, l4, exp_data, J=config["J"], g=config["g"], sigma_narrow=config["sigma_narrow"], sigma_broad=config["sigma_broad"], kappa=config["kappa"])
    
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

    scm.plot_weights()
    
    visualize_connectivity(l4, l23, scm, selected_idx=150, layer='L4', direction='out')
    visualize_connectivity(l4, l23, scm, selected_idx=200, layer='L2/3', direction='in')
    # interactive_connectivity(l4, l23, scm, direction='in')   # 查看各个神经元的感受野 (接收)
    # interactive_connectivity(l4, l23, scm, direction='out')   # 查看各个神经元的感受野 (接收)