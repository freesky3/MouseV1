import numpy as np
from scipy.special import erf
import scipy.integrate as scpint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

param = 5.5

def f(x):
    """
    f(x) is e^x^2 * (1 + erf(x))
    """
    x = np.atleast_1d(x)
    res = np.zeros_like(x, dtype=float)
    
    # Case 1: x > 5.5 (使用大值近似)
    if np.any(x > param):
        raise ValueError("x is too large, which means the integration upper bound is too large")


    # Case 2: x >= param (使用标准公式)
    mask_large = (x >= -param)
    if np.any(mask_large):
        val = x[mask_large]
        res[mask_large] = np.exp(val**2) * (1 + erf(val))
        
    # Case 3: x < param (使用泰勒展开近似)
    mask_small = ~mask_large
    if np.any(mask_small):
        val = x[mask_small]
        val2 = val * val
        res[mask_small] = -1 / (np.sqrt(np.pi) * val) * (1.0 - 0.5 / val2 + 0.75 / (val2 * val2))

    return res


def integrale_vec(min_arr, max_arr):
    """
    Integrate the function f(x) from min to max
    """
    # 确保是 numpy 数组
    min_arr = np.atleast_1d(min_arr)
    max_arr = np.atleast_1d(max_arr)
    res = np.zeros_like(max_arr, dtype=float)

    if np.any(max_arr > param):
        raise ValueError("积分上限过大，这通常表明输入的平均电流 mu 极低")

    # 情况 1：需要数值积分的部分 (使用循环，因为 quad 必须标量输入)
    mask_quad = max_arr <= param
    for i in np.where(mask_quad)[0]:
        res[i] = scpint.quad(f, min_arr[i], max_arr[i], epsabs=1e-4, epsrel=1e-4, limit=100)[0]

    # 情况 2：大值近似部分 (完全向量化，速度极快)
    # # 这里我们利用$$\int_0^M e^{u^2} du \sim \frac{e^{M^2}}{2M}$$，（1+erf(x))在x很大时趋近于2
    # mask_approx = ~mask_quad
    # m = max_arr[mask_approx]
    # res[mask_approx] = 1.0 / m * np.exp(m**2)

    return res

def comp_phi_tab(mu, tau, tau_rp, V_r, theta, sigma_t):
    """
    calculate the firing rate phi(mu) with the Siegert formula:
    parameters:
        mu: input current (mV)
        tau: membrane time constant (ms)
        tau_rp: refractory period (ms)
        sigma: input noise (mV)
    return:
        phi: firing rate (Hz)
    """
    if sigma_t <= 0:
        raise ValueError("Physiological Error: 噪声强度 sigma_t 必须为正值。")
    if tau <= 0 or tau_rp <= 0:
        raise ValueError("Physiological Error: 膜时间常数 tau 和不应期 tau_rp 必须大于 0。")
    if theta <= V_r:
        raise ValueError("Physiological Error: 动作电位阈值 theta 必须严格大于重置电位 V_r。")
    # 确保输入是数组
    mu = np.atleast_1d(mu) 

    if np.any(np.abs(mu) > 100):
        raise ValueError("Physiological Error: 输入电流 mu 绝对值过大(>100)，超出典型视觉皮层神经元的合理生理范围。")
    
    # 计算积分上下限
    min_u = (V_r - mu) / sigma_t
    max_u = (theta - mu) / sigma_t
    
    integral_vals = integrale_vec(min_u, max_u)
    
    # Siegert 公式
    r = 1.0 / (tau_rp + tau * np.sqrt(np.pi) * integral_vals)
    
    # 如果输入原本是标量，返回标量
    if r.size == 1:
        return r.item()
    return r

def tabulate_response(sigma_t, tau_E, tau_I, tau_rp, V_r, theta, mu_tab_max):
    """
    tabulate the response function phi(mu)
    parameters:
        sigma_t: input noise (mV)
        tau_rp: refractory period (ms)
        mu_tab_max: maximum input current (mV)
    return:
        phi_int_E: interpolation function of firing rate of E population (Hz)
        phi_int_I: interpolation function of firing rate of I population (Hz)
    """
    mu_tab = np.linspace(-mu_tab_max, mu_tab_max, int(1000*mu_tab_max))

    phi_tab_E = comp_phi_tab(mu_tab, tau_E, tau_rp, V_r, theta, sigma_t)
    phi_tab_I = comp_phi_tab(mu_tab, tau_I, tau_rp, V_r, theta, sigma_t)

    phi_int_E = interp1d(mu_tab, phi_tab_E, kind='linear')
    phi_int_I = interp1d(mu_tab, phi_tab_I, kind='linear')

    return [phi_int_E, phi_int_I]

if __name__ == "__main__":
    config = {
        "sigma_t": 10,
        "tau_E": 0.02,   # 兴奋性神经元膜时间常数 20ms
        "tau_I": 0.01,   # 抑制性神经元膜时间常数 10ms
        "tau_rp": 2e-3,  # 不应期 2ms (最大放电率 500Hz)
        "V_r": 10,       # 重置电位
        "theta": 20,     # 阈值
        "mu_tab_max": 30 # 扩大评估范围，但限制在抛出 ValueError 的范围(-100~100)以内
    }
    
    phi_E_func, phi_I_func = tabulate_response(**config)
    
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
    phi_E_raw = comp_phi_tab(x_mu, config["tau_E"], config["tau_rp"], config["V_r"], config["theta"], config["sigma_t"])
    phi_I_raw = comp_phi_tab(x_mu, config["tau_I"], config["tau_rp"], config["V_r"], config["theta"], config["sigma_t"])
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
    plt.show()