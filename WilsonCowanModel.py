import numpy as np
import scipy.integrate as scpint


def system_RK45(t, aR, aX, QJ_ij, idx_E, idx_I, idx_X, phi_int_E, phi_int_I, tau_E, tau_I):
    """
    define the differential equation dy/dt = F(y, t)
    Wilson-Cowan model: dr/dt = (-r + phi(mu)) / tau
    parameters:
        t: time (ms)
        aR: activity of internal neurons (E + I)
        aX: activity of external input neurons
        QJ_ij: weight matrix (N_E + N_I, N_E + N_I)
        N_E: number of E neurons (excitatory population)
        N_I: number of I neurons (inhibitory population)
        N_X: number of X neurons (external input population)
        phi_int_E: interpolation function of firing rate of E population
        phi_int_I: interpolation function of firing rate of I population
    return:
        F: derivative of activity
    """
    aALL = np.zeros(QJ_ij.shape[1])
    aALL[idx_E] = aR[idx_E]  # 内部兴奋性神经元 (E)
    aALL[idx_I] = aR[idx_I]  # 内部抑制性神经元 (I)
    aALL[idx_X] = aX # 外部输入神经元 (X)

    # 计算输入 (Total Input)
    # 矩阵乘法：权重矩阵 x 神经元活动向量
    MU_over_tau = np.matmul(QJ_ij, aALL)

    # 循环网络的新状态向量 F (即 dr/dt)
    F = np.zeros(np.shape(aR))

    # E 群体动力学: (-r + phi(input)) / tau
    F[idx_E] = (-aALL[idx_E] + phi_int_E(tau_E * MU_over_tau[idx_E])) / tau_E
    # I 群体动力学
    F[idx_I]  = (-aALL[idx_I] + phi_int_I(tau_I * MU_over_tau[idx_I])) / tau_I
    
    return F


def solve_dynamical_system(aX, QJ_ij, idx_E, idx_I, idx_X, phi_int_E, phi_int_I, tau_E, tau_I, T=None):
    """
    solve the dynamical system with RK45
    parameters:
        aX: activity of external input neurons
        QJ_ij: weight matrix (N_E + N_I, N_E + N_I + N_X)
        idx_E: indices of E neurons
        idx_I: indices of I neurons
        idx_X: indices of X neurons
        phi_int_E: interpolation function of firing rate of E population
        phi_int_I: interpolation function of firing rate of I population
        T: time points (ms)
    return:
        aE: activity of E neurons
        aI: activity of I neurons
    """
    if T is None:
        T=np.arange(0, 100*tau_E, tau_I/3)

    # 求解 IVP (Initial Value Problem)
    # 初始状态设为 0 (aR_t[:,0])
    aR_t = np.zeros((len(idx_E) + len(idx_I), len(T)))
    sol = scpint.solve_ivp(
        system_RK45, 
        [np.min(T), np.max(T)], 
        aR_t[:,0], 
        method='RK45', 
        args=(aX, QJ_ij, idx_E, idx_I, idx_X, phi_int_E, phi_int_I, tau_E, tau_I), 
        t_eval=T
    )

    # 获取解
    aR_t = sol.y  
    aE_t = aR_t[idx_E, :]
    aI_t = aR_t[idx_I, :]
    
    # 检查收敛性 (计算最后 1/3 时间段的标准差均值)
    # 如果系统进入定点 (Fixed Point)，标准差应接近 0
    Convergence_aE = np.mean(np.std(aE_t[:, int(np.shape(aE_t)[1]*2/3):], axis=1))
    Convergence_aI = np.mean(np.std(aI_t[:, int(np.shape(aI_t)[1]*2/3):], axis=1))

    # 计算稳态均值 (Time Averaging)
    aE = np.mean(aE_t[:, int(np.shape(aE_t)[1]*2/3):], axis=1)
    aI = np.mean(aI_t[:, int(np.shape(aI_t)[1]*2/3):], axis=1)
    
    # 重新计算稳态时的输入电流 MU
    aALL = np.zeros(len(idx_E) + len(idx_I) + len(idx_X))
    aALL[idx_E] = aE
    aALL[idx_I] = aI
    aALL[idx_X] = aX
    MU_over_tau = np.matmul(QJ_ij, aALL)
    MU_E = tau_E * MU_over_tau[idx_E]
    MU_I = tau_I * MU_over_tau[idx_I]
    
    Results = [aE, aI, MU_E, MU_I, aE_t, aI_t, Convergence_aE, Convergence_aI]
    return Results


def do_dynamics(QJ_ij, idx_E, idx_I, idx_X, rate_X_of_Theta, phi):
    """
    iterate over all angles Theta, and run the simulation
    parameters:
        QJ: weight matrix (N_E + N_I, N_E + N_I + N_X)
        idx_E: indices of E neurons
        idx_I: indices of I neurons
        idx_X: indices of X neurons
        rate_X_of_Theta: firing rate of external input neurons for each angle
        phi: interpolation functions of firing rate of E and I populations
    return:
        rate_E_of_Theta: firing rate of E neurons for each angle
        rate_I_of_Theta: firing rate of I neurons for each angle
    """
    Theta = np.arange(0, 2*np.pi, np.pi/8.) # 0 到 360度，步长 22.5度
    ntheta = len(Theta)

    ResultsALL = []
    rate_E_of_Theta = np.zeros((len(idx_E), ntheta))
    rate_I_of_Theta = np.zeros((len(idx_I), ntheta))
    
    # 对每个角度进行循环
    for idx_Theta in range(ntheta):
        aX = rate_X_of_Theta[:, idx_Theta] # 获取当前角度的外部输入
        Results = solve_dynamical_system(aX, QJ_ij, idx_E, idx_I, idx_X, phi)
        ResultsALL = ResultsALL + [Results]
        
        aE, aI, MU_E, MU_I, aE_t, aI_t, Convergence_aE, Convergence_aI = Results[:]
        rate_E_of_Theta[:, idx_Theta] = aE
        rate_I_of_Theta[:, idx_Theta] = aI

    return aE_t, rate_E_of_Theta, rate_I_of_Theta