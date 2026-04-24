import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata, interp1d
from src.v1model.default_config import Config
from src.v1model.experimental_data import ExperimentalData
from src.v1model.geometry import L4, L2_3
from src.v1model.SpatialConnectMatrix import SpatialConnectMatrix
from src.v1model.input import L4VisualInput
from src.v1model.NeuronTransfer import tabulate_response
from src.v1model.WilsonCowanModel import WCModel, solve_dynamical_system, do_dynamics


def run_simulation(vis_input, scm, phi_E, phi_I, cfg):
    """在给定的视觉刺激朝向下求解网络演化的常微分方程 (ODE)"""
    print(f"\n--- 开始模拟网络动力学 (刺激朝向 {np.degrees(cfg.theta_stim):.1f}°) ---")
    
    aX_func = vis_input.make_aX_func(cfg.theta_stim)
    
    print("启动 RK45 求解常微分方程 (ODE)...")
    results = solve_dynamical_system(
        aX_func=aX_func, 
        QJ_ij=scm.QJ_ij, 
        idx_E=scm.idx_E, 
        idx_I=scm.idx_I, 
        idx_X=scm.idx_X, 
        phi_int_E=phi_E, 
        phi_int_I=phi_I,
        cfg=cfg
    )
    print("求解完成！")
    return results, aX_func


def plot_results(results, aX_func, l4, l2_3, idx_e, save_path='output/test_result/test_orientation.png'):
    """可视化前馈输入、时间演化以及稳态空间发放图"""
    aE, aI, aE_t, aI_t = results.aE, results.aI, results.aE_t, results.aI_t
    T_eval, conv_aE, conv_aI = results.T_eval, results.conv_aE, results.conv_aI
    
    # 使用 subplots 一次性创建图表骨架，共有 4 个子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # === 图 1: L4 层前馈输入空间图 (取 t=0 时刻的外部输入快照) ===
    a_x_snapshot = aX_func(0.0)
    xX, yX = l4.coords[:, 0], l4.coords[:, 1]
    sc1 = axes[0].scatter(xX, yX, c=a_x_snapshot, cmap='cividis', s=40, edgecolors='k', linewidths=0.5)
    fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04, label="r_X (Hz)")
    axes[0].set_title("L4 Input Population Activity (t=0)")
    axes[0].axis('equal')
    
    # === 图 2: 群体的时间演变过程 ===
    mean_E_t = aE_t.mean(axis=0)
    mean_I_t = aI_t.mean(axis=0)
    
    axes[1].plot(T_eval, mean_E_t, color='red', label='Excitatory (E)', linewidth=2)
    axes[1].plot(T_eval, mean_I_t, color='blue', label='Inhibitory (I)', linewidth=2)
    axes[1].set(xlabel="Time (s)", ylabel="Mean Firing Rate (Hz)", title="Population Dynamics over Time")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # === 图 3: L2/3 内部兴奋性神经元(E)时间平均发放空间图 ===
    xE = [l2_3.coords[i, 0] for i in idx_e]
    yE = [l2_3.coords[i, 1] for i in idx_e]
    sc3 = axes[2].scatter(xE, yE, c=aE, cmap='Reds', s=40, edgecolors='k', linewidths=0.5)
    fig.colorbar(sc3, ax=axes[2], fraction=0.046, pad=0.04, label="r_E (Hz)")
    axes[2].set_title(f"L2/3 E-Neurons Time-Avg\nConv E: {conv_aE:.4f}")
    axes[2].axis('equal')
    
    # === 图 4: 随机选取 6 个神经元 (3E, 3I) 的时间动态 ===
    # 从 E 群体和 I 群体中各随机选取 3 个索引
    N_E, N_I = aE_t.shape[0], aI_t.shape[0]
    sample_E = np.random.choice(N_E, min(3, N_E), replace=False)
    sample_I = np.random.choice(N_I, min(3, N_I), replace=False)
    
    # 绘制选取的 E 神经元
    for i, idx in enumerate(sample_E):
        axes[3].plot(T_eval, aE_t[idx, :], color='salmon', alpha=0.8, 
                     label='Single E' if i==0 else "")
    # 绘制选取的 I 神经元
    for i, idx in enumerate(sample_I):
        axes[3].plot(T_eval, aI_t[idx, :], color='cornflowerblue', alpha=0.8, 
                     label='Single I' if i==0 else "")
                     
    axes[3].set(xlabel="Time (s)", ylabel="Firing Rate (Hz)", title="Single Neuron Dynamics (3E, 3I)")
    axes[3].legend()
    axes[3].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path)

from src.analysis.OSI import get_osi

def plot_orientation_preference_map(l2_3, idx_E, R_E_all, Thetas, save_path='output/test_result/test_orientation_map'):
    """
    根据记录的各个角度的放电率，绘制偏好分布图 (OPM)
    使用 OSI 进行严格的角度判定过滤。
    """
    # 1. 直接使用 OSI 的逻辑 (global-gOSI) 
    osi_vals, pref_angles = get_osi(R_E_all, Thetas, threshold=0.15)
    
    # 2. 依然过滤掉完全不放电的死区 (最大放电率 < 0.1Hz)
    max_rates = np.max(R_E_all, axis=1)
    is_active = max_rates > 0.1
    
    xE = np.array([l2_3.coords[i, 0] for i in idx_E])
    yE = np.array([l2_3.coords[i, 1] for i in idx_E])
    
    plt.figure(figsize=(8, 7))
    
    # 区分有效神经元 (既活跃又有方向选择性) 和 无效神经元 (死寂或没有选择性)
    # np.isnan(pref_angles) 已经包含了 osi < threshold 的条件
    valid_mask = is_active & (~np.isnan(pref_angles))
    invalid_mask = (~valid_mask)
    
    # 画底色 (无效神经元)
    plt.scatter(
        xE[invalid_mask], yE[invalid_mask], 
        color='lightgray', s=20, alpha=0.6, label='Silent or Untuned'
    )
    
    # 画方向图
    if np.sum(valid_mask) > 0:
        # 将角度规范化到 [0, pi) 进行绘图
        plot_angles = np.mod(pref_angles[valid_mask], np.pi)
        
        sc = plt.scatter(
            xE[valid_mask], yE[valid_mask], 
            c=plot_angles, 
            cmap='hsv', 
            vmin=0, vmax=np.pi, 
            s=50, edgecolors='k', linewidths=0.5
        )
        cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
        cbar.set_label("Preferred Orientation (Radians)")
        cbar.set_ticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        cbar.set_ticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])

    plt.title("L2/3 E-Neurons Orientation Preference Map (OSI Filtered)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc='lower right', fontsize=8)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path)

def run_all_orientations_and_plot(cfg, l4, l2_3, vis_input, phi_E, phi_I, scm, save_path='output/test_result/test_orientation_all_map'):
    """主控流程：遍历所有角度并绘制偏好分布图与调谐曲线"""
    Thetas = np.linspace(0, np.pi, cfg.N_theta, endpoint=False)
    print(f"准备开始遍历 {cfg.N_theta} 个刺激角度，这可能需要一些时间...")
    aX_func_of_Theta = [vis_input.make_aX_func(theta) for theta in Thetas]
    
    R_E_all, R_I_all, all_results = do_dynamics(
        QJ_ij=scm.QJ_ij, 
        idx_E=scm.idx_E, 
        idx_I=scm.idx_I, 
        idx_X=scm.idx_X, 
        aX_func_of_Theta=aX_func_of_Theta, 
        phi_int_E=phi_E, 
        phi_int_I=phi_I, 
        cfg=cfg
    )
    
    print("所有角度计算完毕！正在绘制偏好分布图与调谐曲线...")
    # 4. 调用绘图函数绘制 OPM
    plot_orientation_preference_map(l2_3, scm.idx_E, R_E_all, Thetas)
    # 5. 为了清晰展示 Tuning 曲线，我们专门挑选 6 个“调谐最强且偏好角度不同”的 E 神经元
    # 计算每个神经元的调幅 (Amplitude = max - min)
    amplitudes = np.max(R_E_all, axis=1) - np.min(R_E_all, axis=1)
    
    # 找到每个神经元的偏好角度索引
    pref_idx = np.argmax(R_E_all, axis=1)
    
    # 我们希望选出的神经元，其偏好角度尽可能均匀分布在 0~180 度之间
    # 假设测试角度有 16 个，我们每隔几个提取一个角度，找到在那个角度响应最强、且调幅最大的神经元
    target_indices = []
    
    # 均匀选取 6 个目标索引位置去寻找最佳神经元
    ntheta = len(Thetas)
    step = max(1, ntheta // 6)
    for i in range(0, ntheta, step):
        if len(target_indices) >= 6:
            break
            
        # 找到所有偏好当前角度的神经元
        neurons_pref_this_angle = np.where(pref_idx == i)[0]
        
        if len(neurons_pref_this_angle) > 0:
            # 在这些神经元中，选调幅最大的那个
            best_neuron = neurons_pref_this_angle[np.argmax(amplitudes[neurons_pref_this_angle])]
            # 过滤掉几乎不放电的死神经元
            if amplitudes[best_neuron] > 0.5:
                target_indices.append(best_neuron)

    # 如果没有找到足够的不同角度的神经元，就直接从调幅最大的里面补齐
    if len(target_indices) < 6:
        top_tuned = np.argsort(amplitudes)[::-1]
        for idx in top_tuned:
            if idx not in target_indices and amplitudes[idx] > 0.5:
                target_indices.append(idx)
            if len(target_indices) >= 6:
                break

    # 开始画图
    plt.figure(figsize=(10, 6))
    angles_deg = np.degrees(Thetas)
    
    # 使用显眼的颜色和标记区分不同的 E 神经元
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_indices)))
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, idx in enumerate(target_indices):
        pref_angle = angles_deg[pref_idx[idx]]
        plt.plot(angles_deg, R_E_all[idx, :], marker=markers[i%len(markers)], 
                 linestyle='-', color=colors[i], markersize=8, linewidth=2,
                 label=f'E-Neuron ID={idx} (Pref: {pref_angle:.1f}°)')
                 
    plt.title("Strongly Tuned Excitatory Neurons (Diverse Preferences)")
    plt.xlabel("Stimulus Orientation (Degrees)")
    plt.ylabel("Time-Averaged Firing Rate (Hz)")
    plt.xticks(np.arange(0, 181, 45)) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    
    return R_E_all

def analyze_opm_randomness(cfg, l2_3, idx_E, R_E_all, save_path='output/test_result/test_orientation_randomness'):
    """
    分析方向选择性分布的随机性与结构特征。
    包含: 1. 空间自相关性 (Spatial Correlogram)
          2. 频域二维傅里叶变换 (2D FFT Power Spectrum)
    """
    Thetas = np.linspace(0, np.pi, cfg.N_theta, endpoint=False)
    print("\n--- 开始量化分析 OPM 空间分布特征 ---")
    
    # 1. 数据准备与过滤活跃神经元
    pref_idx = np.argmax(R_E_all, axis=1)
    pref_angles = Thetas[pref_idx]
    max_rates = np.max(R_E_all, axis=1)
    active_mask = max_rates > 0.1
    
    if np.sum(active_mask) < 10:
        print("活跃神经元太少，无法进行有意义的空间分析。")
        return

    xE = np.array([l2_3.coords[i, 0] for i in idx_E])[active_mask]
    yE = np.array([l2_3.coords[i, 1] for i in idx_E])[active_mask]
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
    axes[0].plot(bin_centers, mean_diff, 'ro-', markersize=6, linewidth=2, label=r'Mean $\Delta\theta$')
    axes[0].fill_between(bin_centers, mean_diff - std_diff*0.2, mean_diff + std_diff*0.2, 
                         color='red', alpha=0.2, label=r'$\pm 0.2$ Std')
    axes[0].axhline(np.pi/2, color='k', linestyle='--', label=r'Random Expectation ($\pi/2$)')
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
    plt.savefig(save_path)
    print("量化分析完成！")

def main():
    cfg = Config()
    data = ExperimentalData(cfg)
    l4 = L4(cfg, data)
    l2_3 = L2_3(cfg, data)
    vis_input = L4VisualInput(l4, cfg)
    phi_E, phi_I = tabulate_response(cfg)
    scm = SpatialConnectMatrix(l2_3, l4, cfg, data)

    # 运行模拟并绘图
    results, aX_func = run_simulation(vis_input, scm, phi_E, phi_I, cfg)
    plot_results(results, aX_func, l4, l2_3, scm.idx_E)
    
    # 运行所有角度模拟并绘图
    R_E_all = run_all_orientations_and_plot(cfg, l4, l2_3, vis_input, phi_E, phi_I, scm)
    analyze_opm_randomness(cfg, l2_3, scm.idx_E, R_E_all)

if __name__ == "__main__":
    main()