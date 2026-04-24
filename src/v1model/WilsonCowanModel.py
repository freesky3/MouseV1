import numpy as np
import scipy.integrate as scpint
from dataclasses import dataclass
from typing import List

from src.v1model.default_config import Config

# --- 1. 使用 Dataclass 规范返回值结构 ---
@dataclass
class ODEResult:
    aE: np.ndarray       # 时间平均激发率 (N_E,)
    aI: np.ndarray       # 时间平均抑制率 (N_I,)
    aE_t: np.ndarray     # 时间序列 (N_E, T)
    aI_t: np.ndarray     # 时间序列 (N_I, T)
    T_eval: np.ndarray   # 时间点
    conv_aE: float       # 收敛指标
    conv_aI: float       # 收敛指标


# --- 2. 面向对象的模型类 (处理内存预分配) ---
class WCModel:
    def __init__(self, QJ_ij, idx_E, idx_I, idx_X, phi_int_E, phi_int_I, cfg):
        self.QJ_ij = QJ_ij
        self.idx_E = idx_E
        self.idx_I = idx_I
        self.idx_X = idx_X
        self.phi_int_E = phi_int_E
        self.phi_int_I = phi_int_I
        
        self.tau_E = cfg.tau_E
        self.tau_I = cfg.tau_I

        # 预分配
        self._aALL = np.zeros(QJ_ij.shape[1])
        self._F = np.zeros(len(idx_E) + len(idx_I))

    def system_RK45(self, t, aR, aX_func):
        """
        计算 dy/dt = F(y, t)
        """
        self._aALL.fill(0)
        
        # 更新状态
        self._aALL[self.idx_E] = aR[self.idx_E]
        self._aALL[self.idx_I] = aR[self.idx_I]
        self._aALL[self.idx_X] = aX_func(t)

        # 计算总输入
        MU_over_tau = np.matmul(self.QJ_ij, self._aALL)

        # 计算导数
        self._F[self.idx_E] = (-aR[self.idx_E] + self.phi_int_E(self.tau_E * MU_over_tau[self.idx_E])) / self.tau_E
        self._F[self.idx_I] = (-aR[self.idx_I] + self.phi_int_I(self.tau_I * MU_over_tau[self.idx_I])) / self.tau_I
        
        return self._F.copy()


# --- 3. 求解单个外部输入的系统 ---
def solve_dynamical_system(aX_func, QJ_ij, idx_E, idx_I, idx_X, phi_int_E, phi_int_I, cfg, T=None) -> ODEResult:
    """
    solve the dynamical system with RK45
    """
    if T is None:
        T = np.arange(0, 100*cfg.tau_E, cfg.tau_I/3)

    # 1. 实例化模型以获得预分配的内存和上下文
    model = WCModel(QJ_ij, idx_E, idx_I, idx_X, phi_int_E, phi_int_I, cfg)

    # 2. 初始状态设为 0 (一维向量形式)
    y0 = np.zeros(len(idx_E) + len(idx_I))
    
    # 3. 求解 IVP
    # 此时 args 只需要传入 aX_func，因为其他参数都在 model 实例中
    sol = scpint.solve_ivp(
        model.system_RK45, 
        [np.min(T), np.max(T)], 
        y0, 
        method='RK45', 
        args=(aX_func,), 
        t_eval=T
    )

    # 获取解
    aR_t = sol.y  
    T_eval = sol.t
    aE_t = aR_t[idx_E, :]
    aI_t = aR_t[idx_I, :]
    
    # 切片优化：计算最后 1/3 时间段
    idx_23 = int(aR_t.shape[1] * 2 / 3)
    
    aE = np.mean(aE_t[:, idx_23:], axis=1)
    aI = np.mean(aI_t[:, idx_23:], axis=1)

    Convergence_aE = np.mean(np.std(aE_t[:, idx_23:], axis=1))
    Convergence_aI = np.mean(np.std(aI_t[:, idx_23:], axis=1))
    
    # 4. 返回明确类型的对象
    return ODEResult(
        aE=aE, 
        aI=aI, 
        aE_t=aE_t, 
        aI_t=aI_t, 
        T_eval=T_eval, 
        conv_aE=Convergence_aE, 
        conv_aI=Convergence_aI
    )


# --- 4. 纯计算逻辑：遍历所有角度 ---
def do_dynamics(QJ_ij, idx_E, idx_I, idx_X, aX_func_of_Theta, phi_int_E, phi_int_I, cfg):
    """
    仅负责循环计算所有角度的动力学，不包含任何绘图或文件保存逻辑
    """
    ntheta = len(aX_func_of_Theta)
    rate_E_of_Theta = np.zeros((len(idx_E), ntheta))
    rate_I_of_Theta = np.zeros((len(idx_I), ntheta))
    
    # 存储所有角度的完整对象，方便下游随意调取
    all_results: List[ODEResult] = [] 
    
    for idx_Theta in range(ntheta):
        aX_func = aX_func_of_Theta[idx_Theta]
        res = solve_dynamical_system(aX_func, QJ_ij, idx_E, idx_I, idx_X, phi_int_E, phi_int_I, cfg)
        
        # 使用对象的属性访问，非常清晰
        rate_E_of_Theta[:, idx_Theta] = res.aE
        rate_I_of_Theta[:, idx_Theta] = res.aI
        all_results.append(res)

    return rate_E_of_Theta, rate_I_of_Theta, all_results