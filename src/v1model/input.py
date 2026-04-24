import numpy as np
from scipy.interpolate import interp1d

class L4VisualInput:
    """
    initialize geometry.L4
    generate drifting grating L(x,y,t) and compute L4 neuron's firing rate response
    """
    def __init__(self, l4_layer, cfg):
        """
        Args:
            l4_layer (SheetGeometry): L4 object (including neuron coordinates, tuning, etc.)
        """
        self.l4 = l4_layer # geometry.L4 object
        self.N = cfg.N_X # number of L4 neurons
        self.n_side = cfg.L4_n_side # side length of L4 grid
        
        self.fov_size = cfg.fov_size # visual space size, corresponding to the effective calculation area of the receptive field.
        self.sigma = cfg.sigma # Gabor receptive field size.
        self.gamma = cfg.gamma # spatial anisotropy parameter / aspect ratio.
        self.k_base = cfg.k # spatial frequency. For non-tuned neurons, this value will be forced to 0.
        self.psi = cfg.psi # Gabor filter phase.
        self.r0 = cfg.r0 # baseline firing rate.
        self.res = cfg.res # spatial integration grid resolution.

        self.L0 = cfg.L0 # stimulus contrast
        self.epsilon = cfg.epsilon # stimulus contrast modulation depth
        self.omega = cfg.omega # stimulus temporal frequency

        self.visual_gain = cfg.visual_gain
        
        # 1. extract neuron spatial coordinates
        self.x_i = self.l4.coords[:, 0]
        self.y_i = self.l4.coords[:, 1]
        
        # 2. extract preferred orientation (for 'U' non-tuned neurons, set angle to 0 and set k=0 to eliminate tuning)
        self.is_tuned = (self.l4.tunings == 'T')
        self.theta_i = np.nan_to_num(self.l4.pref_dirs, nan=0)
                
        # 3. initialize visual grid space for discrete double integral
        x_vis = np.linspace(-self.fov_size/2, self.fov_size/2, self.res)
        y_vis = np.linspace(-self.fov_size/2, self.fov_size/2, self.res)
        self.X_vis, self.Y_vis = np.meshgrid(x_vis, y_vis)
        self.dx = x_vis[1] - x_vis[0]
        self.dy = y_vis[1] - y_vis[0]
        
        # pre-compute all neurons' Gabor filters F_i(x,y)
        self.F = self._compute_all_gabors()

    def _compute_all_gabors(self):
        """compute all L4 neurons' Gabor filters F_i(x,y) according to the formula"""
        dx = self.X_vis[np.newaxis, :, :]      
        dy = self.Y_vis[np.newaxis, :, :]      
        THETA = self.theta_i[:, np.newaxis, np.newaxis] 

        # receptive field coordinate rotation transformation
        X_prime = dx * np.cos(THETA) + dy * np.sin(THETA)
        Y_prime = -dx * np.sin(THETA) + dy * np.cos(THETA)
        
        # compute Gabor filters
        gaussian = np.exp(-(X_prime**2 + self.gamma * Y_prime**2) / (2 * self.sigma**2))
        
        # for non-tuned ('U') neurons, let frequency k be 0, so that it degenerates into a pure Gaussian filter
        K_eff = np.where(self.is_tuned[:, np.newaxis, np.newaxis], self.k_base, 0.0)
        grating = np.cos(K_eff * X_prime - self.psi)
        
        return gaussian * grating

    def get_drifting_grating(self, theta_stim, t):
        """generate drifting grating L(x,y,t) - formula (4)"""
        XI = self.x_i[:, np.newaxis, np.newaxis]
        YI = self.y_i[:, np.newaxis, np.newaxis]
        X = XI + self.X_vis[np.newaxis, :, :]
        Y = YI + self.Y_vis[np.newaxis, :, :]
        
        phase = X * self.k_base * np.cos(theta_stim) + Y * self.k_base * np.sin(theta_stim) - self.omega * t
        L = self.L0 * (1 + self.epsilon * np.cos(phase))
        return L

    def get_input_at_theta(self, theta_stim, t):
        """
        compute L4 neuron's firing rate r_i^X(t) - formula (3).
        return shape (N,) numpy array, which can be directly sent to L2/3 differential equation model.
        """
        # 1. generate drifting grating
        L = self.get_drifting_grating(theta_stim, t)
        
        # 2. spatial integral operation: sum over (F_i * L) * dx * dy
        integral_val = np.sum(self.F * L, axis=(1, 2)) * self.dx * self.dy
        
        # 3. add baseline and apply ReLU activation function [x]_+
        r_X = self.r0 + integral_val
        return np.maximum(0, r_X) # shape: (N,), N is # of L4 neurons

    def make_aX_func(self, theta_stim, n_samples=400):
        """
        构造含时的外部输入函数 aX(t)，使用数学分析优化极大地加速求值。
        
        把 L0*(1 + eps*cos(A - wt)) 展开为 L0 + L0*eps*(cos(A)*cos(wt) + sin(A)*sin(wt))
        提前积出空间部分，使时间演化计算只需要 O(1)。
        """
        XI = self.x_i[:, np.newaxis, np.newaxis]
        YI = self.y_i[:, np.newaxis, np.newaxis]
        X = XI + self.X_vis[np.newaxis, :, :]
        Y = YI + self.Y_vis[np.newaxis, :, :]
        
        A = X * self.k_base * np.cos(theta_stim) + Y * self.k_base * np.sin(theta_stim)
        
        # Precompute spatial integrals for this grating
        int_base = np.sum(self.F * self.L0, axis=(1, 2)) * self.dx * self.dy
        int_cos = np.sum(self.F * (self.L0 * self.epsilon * np.cos(A)), axis=(1, 2)) * self.dx * self.dy
        int_sin = np.sum(self.F * (self.L0 * self.epsilon * np.sin(A)), axis=(1, 2)) * self.dx * self.dy

        def aX_func(t):
            cos_wt = np.cos(self.omega * t)
            sin_wt = np.sin(self.omega * t)
            integral_val = int_base + int_cos * cos_wt + int_sin * sin_wt
            r_X = self.r0 + integral_val
            return np.maximum(0, r_X) * self.visual_gain
        
        return aX_func