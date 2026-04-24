import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SheetGeometry:
    def __init__(self, n_side, region_size, z_pos):
        # layer geometry parameters
        self.n_side = n_side
        self.N = n_side * n_side
        self.region_size = region_size
        self.z_pos = z_pos
        
        # neurons' properties
        self.coords = self._generate_grid_positions()
        self.types = np.nan
        self.tunings = np.nan
        self.pref_dirs = np.nan

    def _generate_grid_positions(self):
        half_size = self.region_size / 2.0
        x = np.linspace(-half_size, half_size, self.n_side)
        y = np.linspace(-half_size, half_size, self.n_side)
        X, Y = np.meshgrid(x, y)
        coords = np.column_stack((X.ravel(), Y.ravel()))
        return coords # shape: (N, 2)

    def get_distance_matrix(self, periodic):
        delta = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        if periodic:
            abs_delta = np.abs(delta)
            delta = np.minimum(abs_delta, self.region_size - abs_delta)
        return np.sqrt(np.sum(delta**2, axis=2)) # shape: (N, N)

    def get_distance_to(self, other_layer, periodic):
        delta_2d = self.coords[:, np.newaxis, :] - other_layer.coords[np.newaxis, :, :]
        if periodic:
            abs_delta = np.abs(delta_2d)
            delta_2d = np.minimum(abs_delta, self.region_size - abs_delta)
        dist_2d_sq = np.sum(delta_2d**2, axis=2)
        z_diff = self.z_pos - other_layer.z_pos
        return np.sqrt(dist_2d_sq + z_diff**2) # shape: (N1**2, N2**2)

class L4(SheetGeometry):
    def __init__(self, cfg, exp_data):
        super().__init__(cfg.L4_n_side, cfg.L4_region_size, cfg.L4_z_pos)
        self.cfg = cfg
        self.exp_data = exp_data
        self._set_neurons()
        
    
    def _set_neurons(self):
        if self.cfg.all_tuned:
            self.tunings = np.full(self.N, 'T')
        else:
            self.tunings = np.random.choice(['T', 'U'], size=self.N, p=[self.exp_data.pT_X, self.exp_data.pU_X])
        self.pref_dirs = np.full(self.N, np.nan)
        is_T = (self.tunings == 'T')
        self.pref_dirs[is_T] = np.random.choice(self.exp_data.Theta, size=np.sum(is_T))
        

class L2_3(SheetGeometry):
    def __init__(self, cfg, exp_data):
        super().__init__(cfg.L2_3_n_side, cfg.L2_3_region_size, cfg.L2_3_z_pos)
        self.cfg = cfg
        self.exp_data = exp_data
        self._set_neurons()
    
    def _set_neurons(self):
        if self.cfg.random_I:
            self.types = np.random.choice(['E', 'I'], size=self.N, p=[self.exp_data.pE, self.exp_data.pI])
        else:
            # Inhibitory neurons are uniform distributed in the 2D grid
            N_I = self.exp_data.N_I
            n_I_side = int(round(np.sqrt(N_I)))
            self.exp_data.N_I = n_I_side * n_I_side
            self.exp_data.N_E = self.N - self.exp_data.N_I
            self.types = np.full(self.N, 'E')
            row_indices = np.round(np.linspace(0, self.n_side - 1, n_I_side, endpoint=False)).astype(int)
            col_indices = np.round(np.linspace(0, self.n_side - 1, n_I_side, endpoint=False)).astype(int)
            for r in row_indices:
                for c in col_indices:
                    self.types[r * self.n_side + c] = 'I'
        
        self.tunings = np.empty(self.N, dtype='<U1')
        is_I = (self.types == 'I')
        self.tunings[is_I] = 'U' # assume all I neurons are untuned
        is_E = ~is_I
        self.tunings[is_E] = 'T' # assume all E neurons are tuned
        self.pref_dirs = np.full(self.N, np.nan)