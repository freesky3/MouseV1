import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# 假设以下模块在外部文件中已正确实现
from experimental_data import ExperimentalData
from geometry import L4, L2_3
from SpatialConnectMatrix import SpatialConnectMatrix
from input import L4VisualInput
from NeuronTransfer import tabulate_response
from WilsonCowanModel import solve_dynamical_system

from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata

@dataclass
class SimulationConfig:
    """管理模拟过程中的所有超参数与网络配置"""
    # L4 参数
    L4_n_side: int = 20
    L4_region_size: float = 1.0
    L4_z_pos: float = 0.0
    
    # L2/3 参数
    L2_3_region_size: float = 1.0
    L2_3_z_pos: float = 0

    # 突触连接与网络权重参数
    p_EE: float = 0.1
    J: float = 3.0
    g: float = 1.1
    sigma_narrow: float = 0.05
    sigma_broad: float = 0.15
    kappa: float = 0.85

    # 视觉刺激参数 (Gabor 滤波器)
    fov_size: float = 1.0     # 视场大小
    sigma: float = 0.1        # 空间高斯核标准差
    gamma: float = 1.0        # 空间各向异性参数 / 纵横比
    k: float = np.pi * 4      # 空间频率
    psi: float = 0.0          # 滤波器相位
    r0: float = 0.0           # 基线放电率
    res: int = 300             # 空间积分的网格分辨率
    theta_stim: float = np.pi / 4 # 刺激朝向

    t_start: float = 0.0
    L0: float = 1.0
    epsilon: float = 1.0
    omega: float = 2 * np.pi

    visual_gain: float = 400.0

    # 神经元群体动力学参数
    tau_E: float = 0.02       # 兴奋性神经元时间常数 (20ms)
    tau_I: float = 0.01       # 抑制性神经元时间常数 (10ms)
    theta: float = 20.0       # 发放阈值 (mV)
    V_r: float = 10.0         # 复位电位 (mV)


def generate_phase_invariant_input(vis_input, theta_stim, omega, time_steps=30):
    """
    生成稳态网络需要的输入 current。
    消除光栅时间相位带来的空间不均匀性，取一个漂移周期内的最大放电响应。
    """
    cycle_duration = 2 * np.pi / omega
    t_samples = np.linspace(0, cycle_duration, time_steps, endpoint=False)
    
    # 使用列表推导式精简代码
    rates = [vis_input.get_input_at_theta(theta_stim, t=t, omega=omega) for t in t_samples]
    return np.max(rates, axis=0)  # shape: (N_X,)


def run_simulation(vis_input, scm, phi_E, phi_I, cfg: SimulationConfig):
    """在给定的视觉刺激朝向下求解网络演化的常微分方程 (ODE)"""
    print(f"\n--- 开始模拟网络动力学 (刺激朝向 {np.degrees(cfg.theta_stim):.1f}°) ---")
    
    a_x = generate_phase_invariant_input(vis_input, cfg.theta_stim)
    a_x = a_x * SimulationConfig.visual_gain
    
    print("启动 RK45 求解常微分方程 (ODE)...")
    results = solve_dynamical_system(
        aX=a_x, 
        QJ_ij=scm.QJ_ij, 
        idx_E=scm.idx_E, 
        idx_I=scm.idx_I, 
        idx_X=scm.idx_X, 
        phi_int_E=phi_E, 
        phi_int_I=phi_I,
        tau_E=cfg.tau_E,
        tau_I=cfg.tau_I,
    )
    print("求解完成！")
    return results, a_x


def plot_results(results, a_x, l4, l2_3, idx_e):
    """可视化前馈输入、时间演化以及稳态空间发放图"""
    aE, aI, MU_E, MU_I, aE_t, aI_t, conv_aE, conv_aI = results
    
    # 使用 subplots 一次性创建图表骨架，更加优雅
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # === 图 1: L4 层前馈输入空间图 ===
    xX, yX = l4.coords[:, 0], l4.coords[:, 1]
    sc1 = axes[0].scatter(xX, yX, c=a_x, cmap='cividis', s=40, edgecolors='k', linewidths=0.5)
    fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04, label="r_X (Hz)")
    axes[0].set_title("L4 Input Population Activity")
    axes[0].axis('equal')
    
    # === 图 2: 群体稳态的时间演变过程 ===
    mean_E_t = aE_t.mean(axis=0)
    mean_I_t = aI_t.mean(axis=0)
    time_axis = np.linspace(0, len(mean_E_t) * 0.01, len(mean_E_t))
    
    axes[1].plot(time_axis, mean_E_t, color='red', label='Excitatory (E)', linewidth=2)
    axes[1].plot(time_axis, mean_I_t, color='blue', label='Inhibitory (I)', linewidth=2)
    axes[1].set(xlabel="Time steps (~ms)", ylabel="Mean Firing Rate (Hz)", title="Population Dynamics over Time")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # === 图 3: L2/3 内部兴奋性神经元(E)稳态发放空间图 ===
    xE = [l2_3.neurons[i].pos[0] for i in idx_e]
    yE = [l2_3.neurons[i].pos[1] for i in idx_e]
    sc3 = axes[2].scatter(xE, yE, c=aE, cmap='Reds', s=40, edgecolors='k', linewidths=0.5)
    fig.colorbar(sc3, ax=axes[2], fraction=0.046, pad=0.04, label="r_E (Hz)")
    axes[2].set_title(f"L2/3 E-Neurons Steady State\nConv E: {conv_aE:.4f}")
    axes[2].axis('equal')
    
    plt.tight_layout()
    plt.show()

def plot_orientation_preference_map(l2_3, idx_E, R_E_all, Theta):
    """
    根据记录的各个角度的放电率，绘制偏好分布图 (OPM)
    R_E_all shape: (N_E, ntheta)
    """
    # 1. 找到每个神经元响应最大的角度索引
    pref_idx = np.argmax(R_E_all, axis=1)
    pref_angles = Theta[pref_idx]
    
    # 2. 提取最大放电率，用于过滤那些完全不放电的"死"神经元
    max_rates = np.max(R_E_all, axis=1)
    
    # 我们只绘制最大放电率大于 0.1 Hz 的活跃神经元
    active_mask = max_rates > 0.1
    
    xE = np.array([l2_3.neurons[i].pos[0] for i in idx_E])
    yE = np.array([l2_3.neurons[i].pos[1] for i in idx_E])
    
    # 3. 绘图
    plt.figure(figsize=(8, 7))
    
    # 使用 'hsv' 色谱，因为它是一个完美的环形色谱，非常适合表示 0~2pi 的角度
    sc = plt.scatter(
        xE[active_mask], yE[active_mask], 
        c=pref_angles[active_mask], 
        cmap='hsv', 
        vmin=0, vmax=2*np.pi, 
        s=50, edgecolors='k', linewidths=0.5
    )
    
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label("Preferred Orientation (Radians)")
    
    # 格式化一下 colorbar 的刻度，让人看得更直观
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    
    plt.title("L2/3 E-Neurons Orientation Preference Map")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def run_all_orientations_and_plot():
    """主控流程：遍历所有角度并绘制偏好分布图"""
    # 1. 采用我们修正过的参数配置
    cfg = SimulationConfig()
    cfg.L2_3_z_pos = 0.0      # 消除 Z 轴对距离核的扁平化影响
    cfg.visual_gain = 400.0   # 放大微弱的视觉空间积分前馈
    cfg.sigma_narrow = 0.05   # 修正后的感受野参数
    cfg.sigma_broad = 0.15
    cfg.g = 1.1               # 适度的抑制主导
    
    np.random.seed(42)
    data = ExperimentalData(L4_n_side**2, cfg.p_EE)
    
    l4 = L4(cfg.L4_n_side, cfg.L4_region_size, cfg.L4_z_pos, data.pT_X, data.pU_X, data.Theta)
    l2_3 = L2_3(data.L2_3_n_side, cfg.L2_3_region_size, cfg.L2_3_z_pos, data.PE, data.PI)

    vis_input = L4VisualInput(l4, cfg.fov_size, cfg.sigma, cfg.gamma, cfg.k, cfg.psi, cfg.r0, cfg.res)
    phi_E, phi_I = tabulate_response()
    
    scm = SpatialConnectMatrix(l2_3, l4, data, cfg.J, cfg.g, cfg.sigma_narrow, cfg.sigma_broad, cfg.kappa)

    # 2. 准备接收数据的矩阵
    ntheta = len(data.Theta)
    N_E = len(scm.idx_E)
    R_E_all = np.zeros((N_E, ntheta))
    
    print(f"准备开始遍历 {ntheta} 个刺激角度，这可能需要一些时间...")
    
    # 3. 遍历每个刺激角度
    for i, theta_stim in enumerate(data.Theta):
        print(f"--> [ {i+1} / {ntheta} ] 正在计算刺激朝向 {np.degrees(theta_stim):.1f}° ...")
        
        # 获取输入并施加我们之前讨论的视觉增益
        a_x = generate_phase_invariant_input(vis_input, theta_stim)
        a_x = a_x * cfg.visual_gain
        
        # 求解动力学 ODE
        results = solve_dynamical_system(
            aX=a_x, QJ_ij=scm.QJ_ij, 
            idx_E=scm.idx_E, idx_I=scm.idx_I, idx_X=scm.idx_X, 
            phi_int_E=phi_E, phi_int_I=phi_I,
            tau_E=cfg.tau_E, tau_I=cfg.tau_I
        )
        
        # 提取当前角度下，兴奋性神经元(E)的稳态放电率 (aE)
        aE = results[0] 
        R_E_all[:, i] = aE
        
    print("所有角度计算完毕！正在绘制偏好分布图...")
    # 4. 调用绘图函数
    plot_orientation_preference_map(l2_3, scm.idx_E, R_E_all, data.Theta)

def analyze_opm_randomness(l2_3, idx_E, R_E_all, Theta):
    """
    分析方向选择性分布的随机性与结构特征。
    包含: 1. 空间自相关性 (Spatial Correlogram)
          2. 频域二维傅里叶变换 (2D FFT Power Spectrum)
    """
    print("\n--- 开始量化分析 OPM 空间分布特征 ---")
    
    # 1. 数据准备与过滤活跃神经元
    pref_idx = np.argmax(R_E_all, axis=1)
    pref_angles = Theta[pref_idx]
    max_rates = np.max(R_E_all, axis=1)
    active_mask = max_rates > 0.1
    
    if np.sum(active_mask) < 10:
        print("活跃神经元太少，无法进行有意义的空间分析。")
        return

    xE = np.array([l2_3.neurons[i].pos[0] for i in idx_E])[active_mask]
    yE = np.array([l2_3.neurons[i].pos[1] for i in idx_E])[active_mask]
    angles = pref_angles[active_mask]
    
    # ==========================================
    # 分析 1: 空间自相关图 (Spatial Correlogram)
    # ==========================================
    print("正在计算空间自相关性...")
    coords = np.column_stack((xE, yE))
    
    # 计算所有神经元对之间的欧式距离
    dist_matrix = squareform(pdist(coords))
    
    # 计算所有神经元对之间的角度差 (考虑周期性 0~2pi)
    # 计算最短角度差：min(|a-b|, 2pi - |a-b|)
    angle_diff = np.abs(angles[:, None] - angles[None, :])
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    
    # 提取上三角矩阵（消除自身和重复计算）
    triu_idx = np.triu_indices_from(dist_matrix, k=1)
    distances = dist_matrix[triu_idx]
    differences = angle_diff[triu_idx]
    
    # 距离分箱 (Binning)
    num_bins = 40
    max_dist = np.max(distances) / 2.0  # 只看最大距离的一半以保证统计可靠性
    bins = np.linspace(0, max_dist, num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    mean_diff = np.zeros(num_bins - 1)
    std_diff = np.zeros(num_bins - 1)
    
    for i in range(num_bins - 1):
        mask = (distances >= bins[i]) & (distances < bins[i+1])
        if np.any(mask):
            mean_diff[i] = np.mean(differences[mask])
            std_diff[i] = np.std(differences[mask])
        else:
            mean_diff[i] = np.nan
            
    # ==========================================
    # 分析 2: 二维傅里叶变换 (2D FFT)
    # ==========================================
    print("正在计算 2D FFT 功率谱...")
    # 将角度映射为复数场 Z = exp(i * theta)
    # 如果你的角度是 0~pi，通常乘以 2；如果是 0~2pi 漂移光栅，直接使用即可
    Z = np.exp(1j * angles)
    
    # 因为神经元散布在空间中，需要将其插值到规则的 2D 网格上才能进行 FFT
    grid_res = 100
    grid_x, grid_y = np.mgrid[min(xE):max(xE):complex(grid_res), 
                              min(yE):max(yE):complex(grid_res)]
    
    # 使用 nearest 插值填补空隙
    Z_grid = griddata(coords, Z, (grid_x, grid_y), method='nearest')
    
    # 计算 2D FFT 并将零频移到中心
    fft_Z = np.fft.fftshift(np.fft.fft2(Z_grid))
    power_spectrum = np.abs(fft_Z)**2
    
    # 消除直流分量 (中心点) 以便更好地观察空间频率特征
    center = grid_res // 2
    power_spectrum[center, center] = 0 
    
    # ==========================================
    # 绘图可视化
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图 1: 空间自相关
    axes[0].plot(bin_centers, mean_diff, 'ro-', markersize=6, linewidth=2, label='Mean $\Delta\\theta$')
    axes[0].fill_between(bin_centers, mean_diff - std_diff*0.2, mean_diff + std_diff*0.2, 
                         color='red', alpha=0.2, label='$\pm 0.2$ Std')
    axes[0].axhline(np.pi/2, color='k', linestyle='--', label='Random Expectation ($\pi/2$)')
    axes[0].set_xlabel('Spatial Distance')
    axes[0].set_ylabel('Mean Angle Difference (Radians)')
    axes[0].set_title('Spatial Correlogram')
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)
    
    # 图 2: 2D FFT 功率谱
    extent = [-0.5, 0.5, -0.5, 0.5] # 归一化频率坐标
    im = axes[1].imshow(np.log1p(power_spectrum.T), extent=extent, 
                        origin='lower', cmap='viridis')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Log Power')
    axes[1].set_xlabel('Spatial Frequency X')
    axes[1].set_ylabel('Spatial Frequency Y')
    axes[1].set_title('2D FFT Power Spectrum (Log Scale)')
    
    plt.tight_layout()
    plt.show()
    print("量化分析完成！")


def main(): 
    # 初始化配置对象并设置随机种子
    cfg = SimulationConfig()
    np.random.seed(42)
    
    # 加载底层几何与参数数据
    data = ExperimentalData(cfg.L4_n_side**2, cfg.p_EE)
    
    # 构建 L4 与 L2/3 空间层
    l4 = L4(
        n_side=cfg.L4_n_side, region_size=cfg.L4_region_size, z_pos=cfg.L4_z_pos, 
        pT_X=data.pT_X, pU_X=data.pU_X, Theta=data.Theta
    )
    l2_3 = L2_3(
        n_side=data.L2_3_n_side, region_size=cfg.L2_3_region_size, z_pos=cfg.L2_3_z_pos, 
        pE=data.PE, pI=data.PI
    )

    # 实例化视觉输入模块
    vis_input = L4VisualInput(
        l4, fov_size=cfg.fov_size, sigma=cfg.sigma, gamma=cfg.gamma, 
        k=cfg.k, psi=cfg.psi, r0=cfg.r0, res=cfg.res, t_start=cfg.t_start, 
        L0=cfg.L0, epsilon=cfg.epsilon, omega=cfg.omega
    )

    # 获取传递函数并计算空间连接矩阵
    phi_E, phi_I = tabulate_response(
        sigma_t=10, tau_E=cfg.tau_E, tau_I=cfg.tau_I, 
        tau_rp=2e-3, V_r=cfg.V_r, theta=cfg.theta, mu_tab_max=30
    )
    scm = SpatialConnectMatrix(
        l23_layer=l2_3, l4_layer=l4, config_data=data, J=cfg.J, g=cfg.g, 
        sigma_narrow=cfg.sigma_narrow, sigma_broad=cfg.sigma_broad, kappa=cfg.kappa
    )

    # 运行模拟并绘图
    results, a_x = run_simulation(vis_input, scm, phi_E, phi_I, cfg)
    plot_results(results, a_x, l4, l2_3, scm.idx_E)
    run_all_orientations_and_plot()
    analyze_opm_randomness(l2_3, scm.idx_E, R_E_all, data.Theta)

if __name__ == "__main__":
    main()