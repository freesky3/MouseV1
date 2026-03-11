import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

from geometry import L4

# 假设在实际运行中，你可以这样导入你的模块：
# from geometry import L4, config

class L4VisualInput:
    """
    接收 geometry.L4 对象的视觉输入层模拟。
    负责生成漂移光栅 L(x,y,t) 并计算 L4 神经元的前馈放电率响应。
    """
    def __init__(self, l4_layer, fov_size, sigma, gamma, k, psi, r0, res, t_start, L0, epsilon, omega):
        """
        Args:
            l4_layer (SheetGeometry): 已经实例化的 L4 对象 (包含神经元坐标、调谐等信息)。
            fov_size (float): 视觉空间大小，对应于感受野的有效计算区域。
            sigma (float): Gabor 感受野的大小。
            gamma (float): 空间各向异性参数 / 纵横比。
            k (float): 空间频率。对于非调谐神经元，此值将强制为0。
            psi (float): Gabor 滤波器的相位。
            r0 (float): 基线放电率。
            res (int): 空间积分的网格分辨率。
        """
        self.l4 = l4_layer
        self.fov_size = fov_size
        self.N = l4_layer.N
        self.n_side = l4_layer.n_side
        
        self.sigma = sigma
        self.gamma = gamma
        self.k_base = k
        self.psi = psi
        self.r0 = r0
        self.res = res

        self.t_start = t_start
        self.L0 = L0
        self.epsilon = epsilon
        self.omega = omega
        
        # 1. 提取神经元空间坐标
        self.x_i = self.l4.coords[:, 0]
        self.y_i = self.l4.coords[:, 1]
        
        # 2. 提取偏好朝向 (对于 'U' 非调谐神经元，设角度为 0，并通过设置 k=0 消除调谐)
        self.theta_i = np.zeros(self.N)
        self.is_tuned = np.zeros(self.N, dtype=bool)
        
        for i, nrn in enumerate(self.l4.neurons):
            if nrn.tuning == 'T' and nrn.pref_dir is not None:
                self.theta_i[i] = nrn.pref_dir
                self.is_tuned[i] = True
                
        # 3. 初始化视觉网格空间以进行离散双重积分
        x_vis = np.linspace(-fov_size/2, fov_size/2, self.res)
        y_vis = np.linspace(-fov_size/2, fov_size/2, self.res)
        self.X_vis, self.Y_vis = np.meshgrid(x_vis, y_vis)
        self.dx = x_vis[1] - x_vis[0]
        self.dy = y_vis[1] - y_vis[0]
        
        # 预计算所有神经元的 Gabor 滤波器 F_i(x,y)
        self.F = self._compute_all_gabors()

    def _compute_all_gabors(self):
        """根据公式预计算所有 L4 神经元的滤波器张量"""
        X = self.X_vis[np.newaxis, :, :]      
        Y = self.Y_vis[np.newaxis, :, :]      
        XI = self.x_i[:, np.newaxis, np.newaxis] 
        YI = self.y_i[:, np.newaxis, np.newaxis] 
        THETA = self.theta_i[:, np.newaxis, np.newaxis] 

        # 感受野坐标旋转变换
        X_prime = (X - XI) * np.cos(THETA) + (Y - YI) * np.sin(THETA)
        Y_prime = -(X - XI) * np.sin(THETA) + (Y - YI) * np.cos(THETA)
        
        # 计算 Gabor 滤波器
        gaussian = np.exp(-(X_prime**2 + self.gamma * Y_prime**2) / (2 * self.sigma**2))
        
        # 对于非调谐 ('U') 神经元，让频率 k 为 0，使其退化为纯高斯滤波器
        K_eff = np.where(self.is_tuned[:, np.newaxis, np.newaxis], self.k_base, 0.0)
        grating = np.cos(K_eff * X_prime - self.psi)
        
        return gaussian * grating

    def get_drifting_grating(self, theta_stim, t):
        """生成漂移光栅的亮度场 L(x,y,t) - 公式 (4)"""
        phase = self.X_vis * self.k_base * np.cos(theta_stim) + self.Y_vis * self.k_base * np.sin(theta_stim) - self.omega * t
        L = self.L0 * (1 + self.epsilon * np.cos(phase))
        return L

    def get_input_at_theta(self, theta_stim, t):
        """
        计算 L4 神经元的放电率 r_i^X(t) - 公式 (3)。
        返回 shape 为 (N,) 的 numpy 数组，可直接送入 L2/3 微分方程模型。
        """
        # 1. 产生光栅刺激
        L = self.get_drifting_grating(theta_stim, t)
        
        # 2. 空间积分运算: sum over (F_i * L) * dx * dy
        integral_val = np.sum(self.F * L[np.newaxis, :, :], axis=(1, 2)) * self.dx * self.dy
        
        # 3. 添加基线并运用 ReLU 激活函数 [x]_+
        r_X = self.r0 + integral_val
        return np.maximum(0, r_X) # shape: (N,), N is # of L4 neurons


    def plot_input(self, theta_stim):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title("Drifting Grating Stimulus $L(x,y,t)$")
        ax.axis("off")

        # 获取初始帧并设置 imshow
        L_initial = self.get_drifting_grating(theta_stim=theta_stim, t=self.t_start)

        # 使用 extent 将像素坐标映射到实际的物理坐标 [-fov_size/2, fov_size/2]
        fov = self.fov_size
        img = ax.imshow(L_initial, cmap='gray', vmin=0, vmax=2, 
                        extent=[-fov/2, fov/2, -fov/2, fov/2])

        # 动画更新函数
        def update(frame):
            # 根据帧数计算当前时间 t (可以根据 omega 调整时间步长)
            input_t = self.t_start + frame * 0.05 
            L_t = self.get_drifting_grating(theta_stim=theta_stim, t=input_t)
            img.set_data(L_t)
            return [img]

        # 创建动画: 50帧，每帧间隔 50 毫秒
        anim = FuncAnimation(fig, update, frames=50, interval=50, blit=True)

        # 关闭静态图，防止 Notebook 渲染两次
        plt.close(fig)
        display(HTML(anim.to_jshtml()))

    def plot_gabor_rf_overlay(self, theta_stim, num_samples=40):
        """
        在输入光栅背景上叠加显示 Gabor 感受野的位置、大小和偏好朝向。
        
        Args:
            theta_stim: 背景光栅的刺激朝向。
            num_samples: 绘制的神经元数量。由于神经元可能很多（如400个），
                         全部绘制会互相遮挡，默认随机抽取部分进行可视化。
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Gabor Receptive Fields Overlay (Stim: {np.degrees(theta_stim):.1f}°)")

        # 1. 获取并绘制 t=0 时的背景光栅
        L_bg = self.get_drifting_grating(theta_stim=theta_stim, t=self.t_start)
        fov = self.fov_size
        
        # 使用 origin='lower' 确保 y 轴方向与实际坐标系一致
        ax.imshow(L_bg, cmap='gray', vmin=0, vmax=self.L0 * (1 + self.epsilon),
                  extent=[-fov/2, fov/2, -fov/2, fov/2], origin='lower', alpha=0.6)

        # 2. 随机采样神经元以防画面过于拥挤
        if num_samples is None or num_samples >= self.N:
            indices = np.arange(self.N)
        else:
            indices = np.random.choice(self.N, num_samples, replace=False)

        # 3. 绘制 Gabor 感受野
        for i in indices:
            x, y = self.x_i[i], self.y_i[i]
            is_tuned = self.is_tuned[i]
            theta = self.theta_i[i]

            # 高斯包络: exp(-(x'^2 + gamma * y'^2) / (2 * sigma^2))
            # 1-sigma 边界对应的椭圆宽度和高度
            width = 2 * self.sigma
            height = 2 * self.sigma / np.sqrt(self.gamma) if self.gamma > 0 else 2 * self.sigma
            
            angle_deg = np.degrees(theta)
            color = 'crimson' if is_tuned else 'dodgerblue'

            # 添加感受野包络边界（椭圆）
            ellipse = patches.Ellipse((x, y), width, height, angle=angle_deg, 
                                      edgecolor=color, facecolor='none', lw=2, alpha=0.8)
            ax.add_patch(ellipse)

            if is_tuned:
                # 绘制一条线段指示偏好条纹的方向 (垂直于 k 矢量)
                # 条纹方向是 theta + 90 度
                stripe_angle = theta + np.pi / 2
                dx = (height / 2) * np.cos(stripe_angle)
                dy = (height / 2) * np.sin(stripe_angle)
                ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=1.5, alpha=0.9)
            else:
                # 非调谐神经元用十字中心标记
                ax.plot(x, y, marker='+', color=color, markersize=8)

        ax.set_xlim(-fov/2, fov/2)
        ax.set_ylim(-fov/2, fov/2)
        ax.set_xlabel("X Position (Cortical Space)")
        ax.set_ylabel("Y Position (Cortical Space)")
        
        # 添加图例
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='crimson', lw=2, label='Tuned (T) RF'),
            Line2D([0], [0], marker='+', color='dodgerblue', lw=0, markersize=8, label='Untuned (U) RF')
        ]
        ax.legend(handles=custom_lines, loc='upper right', framealpha=0.9)

        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()

    def plot_response(self, theta_stim):
        # 获取 t=0 时刻，网络对指定朝向光栅的响应
        r_X = self.get_input_at_theta(theta_stim=theta_stim, t=self.t_start)

        plt.figure(figsize=(8, 6))

        # 使用散点图，c参数传入放电率，cmap选择一个显眼的热力图颜色
        scatter = plt.scatter(self.x_i, self.y_i, 
                            c=r_X, cmap='magma', s=80, 
                            edgecolors='gray', linewidth=0.5)

        # 添加颜色条和标签
        cbar = plt.colorbar(scatter)
        cbar.set_label('Firing Rate (Hz)', rotation=270, labelpad=15)

        plt.title(f"L4 Population Response (Stim: {np.degrees(theta_stim):.1f}°, $t = {self.t_start}$)")
        plt.xlabel("X Position (Cortical Space)")
        plt.ylabel("Y Position (Cortical Space)")

        # 保持XY轴比例一致，更符合真实的空间映射
        plt.axis('equal') 
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def plot_phase_invariant_response(self, theta_stim, time_steps):
        """
        绘制稳态网络所需的相位不变响应。
        计算并展示给定刺激朝向下，漂移光栅一个完整周期内的最大放电率分布。
        """
        # 1. 计算一个周期内的最大放电响应
        cycle_duration = 2 * np.pi / self.omega
        t_samples = np.linspace(0, cycle_duration, time_steps, endpoint=False)
        
        # 遍历周期内的各个时间步，计算放电率
        rates = [self.get_input_at_theta(theta_stim, t) for t in t_samples]
        
        # 沿时间轴 (axis=0) 取最大值，得到每个神经元的相位不变响应，shape: (N_X,)
        r_X_max = np.max(rates, axis=0) 

        # 2. 可视化最大响应
        plt.figure(figsize=(8, 6))

        # 使用散点图，c参数传入最大放电率 r_X_max
        scatter = plt.scatter(self.x_i, self.y_i, 
                              c=r_X_max, cmap='magma', s=80, 
                              edgecolors='gray', linewidth=0.5)

        # 添加颜色条和标签
        cbar = plt.colorbar(scatter)
        cbar.set_label('Max Firing Rate (Hz)', rotation=270, labelpad=15)

        # 更新标题以反映这是周期内的最大响应
        plt.title(f"L4 Phase-Invariant Max Response (Stim: {np.degrees(theta_stim):.1f}°)")
        plt.xlabel("X Position (Cortical Space)")
        plt.ylabel("Y Position (Cortical Space)")

        # 保持XY轴比例一致，更符合真实的空间映射
        plt.axis('equal') 
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def plot_all_neurons_firing_rate_over_cycle(self, theta_stim, time_steps):
        """
        绘制 L4 层所有神经元在一个漂移光栅周期内的放电率变化。
        """
        # 1. 计算一个完整周期的时间长度
        cycle_duration = 2 * np.pi / self.omega
        # 生成时间采样点
        t_samples = np.linspace(0, cycle_duration, time_steps, endpoint=False)
        
        # 2. 计算每个时间步的放电率
        # rates 的形状将是 (time_steps, N)
        rates = np.array([self.get_input_at_theta(theta_stim, t) for t in t_samples])
        num_samples = 20
        sample_indices = np.random.choice(self.N, num_samples, replace=False)
        
        # 3. 可视化
        plt.figure(figsize=(10, 6))
        
        # 绘制所有神经元的放电率曲线
        # 因为可能有几百个神经元，使用较低的 alpha 值防止画面糊死
        plt.plot(t_samples, rates[:, sample_indices], color='steelblue', alpha=0.5)
        
        # 计算并绘制种群平均放电率（可选，用粗红线表示）
        mean_rates = np.mean(rates, axis=1)
        plt.plot(t_samples, mean_rates, color='crimson', linewidth=2.5, label='Population Mean')

        # 装饰图表
        plt.title(f"Firing Rates of All L4 Neurons Over One Cycle (Stim: {np.degrees(theta_stim):.1f}°)")
        plt.xlabel("Time (s)")
        plt.ylabel("Firing Rate (Hz)")
        plt.xlim(0, cycle_duration)
        
        # 避免图例重复（因为 plt.plot 画矩阵会产生 N 个 label）
        handles, labels = plt.gca().get_legend_handles_labels()
        # 只保留平均线的图例，并手动添加一条代表单个神经元的图例
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='steelblue', alpha=0.5, lw=1.5),
                        handles[-1]] # handles[-1] 是均值线
        plt.legend(custom_lines, ['Individual Neurons', 'Population Mean'], loc='upper right')
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    config = {
        "L4_n_side": 20,
        "L4_region_size": 1,
        "L2_3_region_size": 1,
        "L4_z_pos": 0,
        "L2_3_z_pos": 0.5,
        "pT_X": 0.5714,
        "pU_X": 0.4286,
        "Theta": np.linspace(0, np.pi, 16, endpoint=False), 

        "fov_size": 1.0,
        "sigma": 0.1,
        "gamma": 1.0,
        "k": np.pi * 4,
        "psi": 0.0,
        "r0": 0.0,
        "res": 300, 

        "t_start": 0.0,
        "L0": 1.0,
        "epsilon": 1.0,
        "omega": 2 * np.pi,

        "theta_stim": np.pi/4, 
        "time_steps": 100, 
    }
    l4 = L4(n_side=config["L4_n_side"], region_size=config["L4_region_size"], z_pos=config["L4_z_pos"], pT_X=config["pT_X"], pU_X=config["pU_X"], Theta=config["Theta"])
    l4_input = L4VisualInput(l4, fov_size=config["fov_size"], sigma=config["sigma"], gamma=config["gamma"], k=config["k"], psi=config["psi"], r0=config["r0"], res=config["res"], t_start=config["t_start"], L0=config["L0"], epsilon=config["epsilon"], omega=config["omega"])
    l4_input.plot_input(theta_stim=config["theta_stim"])
    l4_input.plot_gabor_rf_overlay(theta_stim=config["theta_stim"])
    l4_input.plot_response(theta_stim=config["theta_stim"])
    l4_input.plot_phase_invariant_response(theta_stim=config["theta_stim"], time_steps=config["time_steps"])
    l4_input.plot_all_neurons_firing_rate_over_cycle(theta_stim=config["theta_stim"], time_steps=config["time_steps"])