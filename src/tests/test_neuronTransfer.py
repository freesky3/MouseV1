import numpy as np
import os
import matplotlib.pyplot as plt
from src.v1model.default_config import Config
from src.v1model.NeuronTransfer import tabulate_response, comp_phi_tab, f, integrale_vec

def main():
    save_path = 'output/test_result/test_NeuronTransfer.png'
    cfg = Config()
    phi_E_func, phi_I_func = tabulate_response(cfg)
    
    # 构造用于绘图的 x 轴数据
    x_f = np.linspace(-5, 5, 1000)         # 用于 f(x)
    x_mu = np.linspace(-10, 30, 1000)      # 用于 mu 电流
    
    # 开始绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 绘制底层函数 f(x)
    axes[0, 0].plot(x_f, f(x_f), color='purple', lw=2)
    axes[0, 0].set_title(r"Base Function: $f(x) = e^{x^2}(1 + erf(x))$")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 绘制积分项 Integral (固定下限测试)
    # 以从 0 到 x 的积分为例展示积分增长趋势
    test_integrals = integrale_vec(np.zeros_like(x_f), x_f)
    axes[0, 1].plot(x_f, test_integrals, color='teal', lw=2)
    axes[0, 1].set_title(r"Integral Term: $\int_0^x f(u)du$")
    axes[0, 1].set_xlabel("x (Integration Upper Bound)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 绘制真实的 Siegert 激发率 (未插值)
    phi_E_raw = comp_phi_tab(x_mu, tau = cfg.tau_E, cfg = cfg)
    phi_I_raw = comp_phi_tab(x_mu, tau = cfg.tau_I, cfg = cfg)
    axes[1, 0].plot(x_mu, phi_E_raw, label="Excitatory ($\tau_E=20ms$)", color='red')
    axes[1, 0].plot(x_mu, phi_I_raw, label="Inhibitory ($\tau_I=10ms$)", color='blue')
    axes[1, 0].set_title("Siegert Firing Rate (Raw Calculation)")
    axes[1, 0].set_xlabel(r"Input Current $\mu$ (mV)")
    axes[1, 0].set_ylabel(r"Firing Rate $\phi$ (Hz)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 绘制插值函数结果对比 (Interpolated Tabulation)
    y_E_int = phi_E_func(x_mu)
    y_I_int = phi_I_func(x_mu)
    axes[1, 1].plot(x_mu, y_E_int, '--', label="E (Interpolated)", color='darkred')
    axes[1, 1].plot(x_mu, y_I_int, '--', label="I (Interpolated)", color='darkblue')
    axes[1, 1].set_title("Tabulated Response Functions")
    axes[1, 1].set_xlabel(r"Input Current $\mu$ (mV)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()