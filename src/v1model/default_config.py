from dataclasses import dataclass
import numpy as np

exp_data = np.load('data/sample_data.pkl', allow_pickle=True)
eta_I = exp_data["eta_I"] # 0.1805
eta_X = exp_data["eta_X"] # 0.4382

class Config:
    """set the hyper-parameters of the simulation"""
    def __init__(self, L4_n_side: int = 40, g: float = 1.1):
        self.periodic: bool = True
        # ==================== L4 parameters ====================
        self.L4_n_side = L4_n_side
        self.N_X = L4_n_side * L4_n_side
        self.L4_region_size: float = 2.0
        self.L4_z_pos: float = 0.0
        self.all_tuned: bool = True # set neuron in L4 all tuned
    
        # ==================== L2/3 parameters ====================
        # L2_3_n_side is up to the experimental ratio: N_L4 / N_L2/3
        exact_N_E = self.N_X / eta_X
        exact_N_I = exact_N_E * eta_I
        self.L2_3_n_side = int(np.ceil(np.sqrt(exact_N_E + exact_N_I)))
        self.L2_3_region_size: float = 2.0
        self.L2_3_z_pos: float = 0.1
        self.random_I: bool = False # set inhibitory neuron in L2/3 random distribute or uniform distribute
    

        # ==================== Synaptic connection and network weight parameters ====================
        self.p_EE: float = 0.1
        self.J: float = 3.0
        self.g: float = g
        self.sigma_narrow: float = 0.05
        self.sigma_broad: float = 0.15 # 2
        self.kappa: float = 0.85

        # ==================== Visual stimulus parameters (Gabor filter) ====================
        self.fov_size: float = 2.0     # field of view size
        self.sigma: float = 0.1        # spatial Gaussian kernel standard deviation
        self.gamma: float = 1.0        # spatial anisotropy parameter / aspect ratio
        self.k: float = np.pi * 4      # spatial frequency
        self.psi: float = 0.0          # filter phase
        self.r0: float = 0.0 # 3.0           # baseline firing rate
        self.res: int = 300             # spatial integration grid resolution
        self.theta_stim: float = np.pi / 4 # stimulus orientation

        self.t_start: float = 0.0
        self.L0: float = 1.0
        self.epsilon: float = 1.0
        self.omega: float = 2 * np.pi

        self.visual_gain: float = 400.0
        self.N_theta: int = 8

        # 神经元群体动力学参数
        self.sigma_t: float = 10
        self.tau_E: float = 0.02       # 兴奋性神经元时间常数 (20ms)
        self.tau_I: float = 0.01       # 抑制性神经元时间常数 (10ms)
        self.tau_rp: float = 2e-3
        self.theta: float = 20.0       # 发放阈值 (mV)
        self.V_r: float = 10.0         # 复位电位 (mV)
        self.mu_tab_max: float = 30