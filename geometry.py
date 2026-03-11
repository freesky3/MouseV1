import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from experimental_data import ExperimentalData

class Neuron:
    """定义单个神经元的属性"""
    def __init__(self, idx, x, y, z, n_type, tuning, pref_dir=None):
        self.idx = idx          # 在层内的全局索引
        self.pos = (x, y, z)    # 空间三维坐标
        self.n_type = n_type    # 'E' (兴奋), 'I' (抑制), 'X' (外部)
        self.tuning = tuning    # 'T' (调谐), 'U' (非调谐)
        self.pref_dir = pref_dir
        
    def __str__(self):
        dir_str = f", pref_dir: {self.pref_dir:.2f}" if self.pref_dir is not None else ""
        return f"Neuron_{self.idx}({self.n_type}, {self.tuning}{dir_str}), position: {self.pos}"

class SheetGeometry:
    def __init__(self, n_side, region_size, z_pos):
        self.n_side = n_side
        self.N = n_side * n_side
        self.region_size = region_size
        self.z_pos = z_pos
        
        self.coords = self._generate_grid_positions()

    def _generate_grid_positions(self):
        half_size = self.region_size / 2.0
        x = np.linspace(-half_size, half_size, self.n_side)
        y = np.linspace(-half_size, half_size, self.n_side)
        X, Y = np.meshgrid(x, y)
        coords = np.column_stack((X.ravel(), Y.ravel()))
        return coords


    def get_distance_matrix(self, periodic=False):
        # 距离计算仍然完全依赖 self.coords，速度不受 Neuron 对象化的影响
        delta = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        if periodic:
            abs_delta = np.abs(delta)
            delta = np.minimum(abs_delta, self.region_size - abs_delta)
        return np.sqrt(np.sum(delta**2, axis=2)) # shape: (N, N)

    def get_distance_to(self, other_layer, periodic=False):
        delta_2d = self.coords[:, np.newaxis, :] - other_layer.coords[np.newaxis, :, :]
        if periodic:
            abs_delta = np.abs(delta_2d)
            delta_2d = np.minimum(abs_delta, self.region_size - abs_delta)
        dist_2d_sq = np.sum(delta_2d**2, axis=2)
        z_diff = self.z_pos - other_layer.z_pos
        return np.sqrt(dist_2d_sq + z_diff**2) # shape: (N1, N2)

class L4(SheetGeometry):
    def __init__(self, n_side, region_size, z_pos, pT_X, pU_X, Theta):
        super().__init__(n_side, region_size, z_pos)
        self.neurons = self._populate_neurons(pT_X, pU_X, Theta)
    

    def _populate_neurons(self, pT_X, pU_X, Theta):
        types = np.full(self.N, 'X')
        tunings = np.random.choice(['T', 'U'], size=self.N, p=[pT_X, pU_X])
        pref_dirs = np.full(self.N, None, dtype=object)
        is_T = (tunings == 'T')
        pref_dirs[is_T] = np.random.choice(Theta, size=np.sum(is_T))
        
        neuron_list = []
        for i in range(self.N):
            x, y = self.coords[i]
            nrn = Neuron(idx=i, x=x, y=y, z=self.z_pos, n_type=types[i], tuning=tunings[i], pref_dir=pref_dirs[i])
            neuron_list.append(nrn)
            
        return np.array(neuron_list)

class L2_3(SheetGeometry):
    def __init__(self, n_side, region_size, z_pos, pE, pI):
        super().__init__(n_side, region_size, z_pos)
        self.neurons = self._populate_neurons(pE, pI)
    
    def _populate_neurons(self, pE, pI):
        types = np.random.choice(['E', 'I'], size=self.N, p=[pE, pI])
        tunings = np.empty(self.N, dtype=object)
        

        is_I = (types == 'I')
        tunings[is_I] = 'U' # assume all I neurons are untuned
        is_E = ~is_I
        n_E = np.sum(is_E)
        tunings[is_E] = 'T' # assume all E neurons are tuned

        # In this project, we don't give E a pre-dicide tuning possibility
        # tunings[is_E] = np.random.choice(
        #     ['T', 'U'], 
        #     size=n_E, 
        #     p=[config['data'].pT_E, config['data'].pU_E]
        # )
        

        neuron_list = []
        for i in range(self.N):
            x, y = self.coords[i]
            nrn = Neuron(idx=i, x=x, y=y, z=self.z_pos, n_type=types[i], tuning=tunings[i])
            neuron_list.append(nrn)
        return np.array(neuron_list)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_network(l4, l2_3):
    # 将画布稍微加宽，以便自然地容纳右侧的“图例+颜色条”面板
    fig = plt.figure(figsize=(11, 8))
    
    # 【修改1：消除左侧空白】
    # [left, bottom, width, height]
    # left 设为 0.0 或 0.05 可以把 3D 图推向最左侧，占据 75% 的宽度
    ax = fig.add_axes([0.05, 0.1, 0.70, 0.8], projection='3d')

    # ==============================
    # 绘制 L4 层 (Z = 0)
    # ==============================
    l4_coords = np.array([n.pos for n in l4.neurons])
    l4_tunings = np.array([n.tuning for n in l4.neurons])
    
    # L4 非调谐神经元 (U) -> 灰色圆点
    idx_l4_u = l4_tunings == 'U'
    ax.scatter(l4_coords[idx_l4_u, 0], l4_coords[idx_l4_u, 1], l4_coords[idx_l4_u, 2], 
               c='lightgray', marker='o', s=15, alpha=0.5, label='L4 Untuned (X)')

    # L4 调谐神经元 (T) -> 根据 pref_dir 上色
    idx_l4_t = l4_tunings == 'T'
    l4_pref_dirs = np.array([n.pref_dir for n in l4.neurons if n.tuning == 'T'], dtype=float)
    
    # 【修改2：优化标记】将 marker 从 '^' 改为 'o' (圆点)，在密集时更清晰可辨
    sc_l4 = ax.scatter(l4_coords[idx_l4_t, 0], l4_coords[idx_l4_t, 1], l4_coords[idx_l4_t, 2], 
                       c=l4_pref_dirs, cmap='hsv', marker='o', s=25, alpha=0.9, label='L4 Tuned (X)')

    # ==============================
    # 绘制 L2/3 层 (Z = 0.5)
    # ==============================
    l23_coords = np.array([n.pos for n in l2_3.neurons])
    l23_types = np.array([n.n_type for n in l2_3.neurons])

    # L2/3 抑制性神经元 (I) -> 蓝色倒三角
    idx_l23_i = l23_types == 'I'
    ax.scatter(l23_coords[idx_l23_i, 0], l23_coords[idx_l23_i, 1], l23_coords[idx_l23_i, 2], 
               c='blue', marker='v', s=20, alpha=0.7, label='L2/3 Inhibitory (I)')

    # L2/3 兴奋性神经元 (E) -> 红色方块
    idx_l23_e = l23_types == 'E'
    ax.scatter(l23_coords[idx_l23_e, 0], l23_coords[idx_l23_e, 1], l23_coords[idx_l23_e, 2], 
               c='red', marker='s', s=20, alpha=0.7, label='L2/3 Excitatory (E)')

    # ==============================
    # 图表设置与比例调整
    # ==============================
    ax.set_title("3D Visualization of L4 and L2/3 Layers", pad=20)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position (Layer)")
    ax.set_zticks([l4.z_pos, l2_3.z_pos])
    ax.set_zticklabels(['L4', 'L2/3'])
    
    # 强制设置 3D 框的比例
    ax.set_box_aspect((1, 1, 0.4)) 
    
    # 【修改3：减轻视觉杂乱】让 3D 背景面板透明，并调浅网格线
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.8)})
    ax.yaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.8)})
    ax.zaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.8)})

    # ==============================
    # 【修改4：整合右侧面板 (图例 + 水平颜色条)】
    # ==============================
    # 将图例放在 3D 图右侧稍高的位置，并添加标题
    legend = ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.55), 
                       title="Neuron Properties", frameon=True)
    legend.get_title().set_fontweight('bold')
    
    # 在图例正下方，手动划出一块小区域用于绘制水平颜色条
    # 参数: [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.76, 0.45, 0.15, 0.02]) 
    cbar = fig.colorbar(sc_l4, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Preferred Direction (Radians)", fontsize=10)
    
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
    }
    exp_data = ExperimentalData(config["L4_n_side"]**2, config["p_EE"])
    l4 = L4(n_side=config["L4_n_side"], region_size=config["L4_region_size"], z_pos=config["L4_z_pos"], pT_X=config["pT_X"], pU_X=config["pU_X"], Theta=config["Theta"])
    l2_3 = L2_3(n_side=exp_data.L2_3_n_side, region_size=config["L2_3_region_size"], z_pos=config["L2_3_z_pos"], pE=config["pE"], pI=config["pI"])
    visualize_network(l4, l2_3)
    print(f"L4 neurons: {config['L4_n_side']}**2, L2/3 neurons: {exp_data.L2_3_n_side}**2")
    print(f"Ratio of L2/3 over L4: {exp_data.L2_3_n_side**2 / config['L4_n_side']**2}")
    print(f"L4 tuned neurons: {config['L4_n_side']**2 * config['pT_X']}, {config['pT_X']*100}%")
    print(f"L4 untuned neurons: {config['L4_n_side']**2 * config['pU_X']}, {config['pU_X']*100}%")
    print(f"L2/3 excitatory neurons: {exp_data.L2_3_n_side**2 * config['pE']}, {config['pE']*100}%")
    print(f"L2/3 inhibitory neurons: {exp_data.L2_3_n_side**2 * config['pI']}, {config['pI']*100}%")