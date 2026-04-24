import numpy as np

class SpatialConnectMatrix:
    def __init__(self, l23_layer, l4_layer, cfg, exp_data):
        self.l23 = l23_layer
        self.l4 = l4_layer
        self.data = exp_data
        self.periodic = cfg.periodic

        N_L23 = self.l23.N
        N_L4 = self.l4.N
        idx_L23 = np.arange(N_L23)
        idx_L4 = np.arange(N_L23, N_L23 + N_L4)

        is_E = (self.l23.types == 'E')
        is_I = (self.l23.types == 'I')
        self.idx_E = idx_L23[is_E]
        self.idx_I = idx_L23[is_I]
        self.idx_X = idx_L4
        n_E, n_I, n_X = len(self.idx_E), len(self.idx_I), len(self.idx_X)
        
        self.sigma_narrow = cfg.sigma_narrow
        self.sigma_broad = cfg.sigma_broad
        self.kappa = cfg.kappa

        self.J = cfg.J
        self.g = cfg.g

        self.Q = self.sample_matrix()
        self.J_ij = self.sample_J()
        self.QJ_ij = self.Q * self.J_ij
        
    def spatial_kernel(self, distance_matrix):
        dist_sq = distance_matrix ** 2
        narrow_gauss = np.exp(-dist_sq / (2 * self.sigma_narrow**2))
        broad_gauss = np.exp(-dist_sq / (2 * self.sigma_broad**2))
        return self.kappa * narrow_gauss + (1 - self.kappa) * broad_gauss
    
    def _normalize_probabilities(self, S_matrix, target_p):
        """
        使用二分查找寻找最佳缩放系数，
        确保截断后的概率矩阵平均值严格等于 target_p。
        """
        # 如果目标概率为 0，直接返回全 0 矩阵
        if target_p <= 0:
            return np.zeros_like(S_matrix)
            
        low, high = 0.0, 10000.0  # 缩放系数的搜索范围
        best_P = None
        
        for _ in range(30):  # 30次迭代精度已足够高
            mid = (low + high) / 2
            P_temp = np.clip(mid * S_matrix, 0.0, 1.0)
            current_mean = np.mean(P_temp)
            
            if current_mean > target_p:
                high = mid
            else:
                low = mid
            best_P = P_temp
            
        return best_P


    def sample_matrix(self):
        N_L23 = self.l23.N
        N_L4 = self.l4.N
        N_total = N_L23 + N_L4
        
        # 初始化连接矩阵 Q
        Q = np.full((N_L23, N_total), np.nan)
        
        # 1. 提取各类神经元的全局索引
        idx_L23 = np.arange(N_L23)
        idx_L4 = np.arange(N_L23, N_total) # L4 的索引接在 L2/3 之后
        
        idx_E = self.idx_E
        idx_I = self.idx_I
        idx_X = self.idx_X
        local_idx_X = np.arange(N_L4)


        dist_L23 = self.l23.get_distance_matrix(periodic=self.periodic)
        dist_L4_to_L23 = self.l23.get_distance_to(self.l4, periodic=self.periodic) # Shape: (N_L23, N_L4)
        d_EE = dist_L23[np.ix_(idx_E, idx_E)]
        d_EI = dist_L23[np.ix_(idx_E, idx_I)]
        d_EX = dist_L4_to_L23[np.ix_(idx_E, local_idx_X)]

        d_IE = dist_L23[np.ix_(idx_I, idx_E)]
        d_II = dist_L23[np.ix_(idx_I, idx_I)]
        d_IX = dist_L4_to_L23[np.ix_(idx_I, local_idx_X)]
        
        # calculate the spatial kernel for the given distance matrix
        S_EE = self.spatial_kernel(d_EE)
        S_EI = self.spatial_kernel(d_EI)
        S_EX = self.spatial_kernel(d_EX)
        S_IE = self.spatial_kernel(d_IE)
        S_II = self.spatial_kernel(d_II)
        S_IX = self.spatial_kernel(d_IX)

        # normalize the spatial kernel per row to ensure uniform in-degree without periodic boundaries
        if not self.periodic:
            S_EE = S_EE / np.mean(S_EE, axis=1, keepdims=True)
            S_EI = S_EI / np.mean(S_EI, axis=1, keepdims=True)
            S_EX = S_EX / np.mean(S_EX, axis=1, keepdims=True)
            S_IE = S_IE / np.mean(S_IE, axis=1, keepdims=True)
            S_II = S_II / np.mean(S_II, axis=1, keepdims=True)
            S_IX = S_IX / np.mean(S_IX, axis=1, keepdims=True)

        np.fill_diagonal(S_EE, 0.0)
        np.fill_diagonal(S_II, 0.0)


        P_EE_matrix = self._normalize_probabilities(S_EE, self.data.p_EE)
        P_EI_matrix = self._normalize_probabilities(S_EI, self.data.p_EI)
        P_EX_matrix = self._normalize_probabilities(S_EX, self.data.p_EX)
        
        P_IE_matrix = self._normalize_probabilities(S_IE, self.data.p_IE)
        P_II_matrix = self._normalize_probabilities(S_II, self.data.p_II)
        P_IX_matrix = self._normalize_probabilities(S_IX, self.data.p_IX)

        
        
        # E. 采样
        Q[np.ix_(idx_E, idx_E)] = np.random.binomial(1, P_EE_matrix)
        Q[np.ix_(idx_E, idx_I)] = np.random.binomial(1, P_EI_matrix)
        Q[np.ix_(idx_E, idx_X)] = np.random.binomial(1, P_EX_matrix)
        Q[np.ix_(idx_I, idx_E)] = np.random.binomial(1, P_IE_matrix)
        Q[np.ix_(idx_I, idx_I)] = np.random.binomial(1, P_II_matrix)
        Q[np.ix_(idx_I, idx_X)] = np.random.binomial(1, P_IX_matrix)

        return Q

    def sample_J(self):
        #J=3.;g=1.5;
        idx_E = self.idx_E
        idx_I = self.idx_I
        idx_X = self.idx_X

        J_ij=np.full(self.Q.shape, np.nan)

        J_ij[np.ix_(idx_E,idx_E)] = self.J  * np.random.choice(self.data.sampled_J_EE, size=(len(idx_E), len(idx_E)))
        J_ij[np.ix_(idx_E,idx_I)] = -self.J*self.g*np.random.choice(self.data.sampled_J_EI, size=(len(idx_E), len(idx_I)))
        J_ij[np.ix_(idx_E,idx_X)] = self.J  * np.random.choice(self.data.sampled_J_EX, size=(len(idx_E), len(idx_X)))

        J_ij[np.ix_(idx_I,idx_E)] = self.J  * np.random.choice(self.data.sampled_J_IE, size=(len(idx_I), len(idx_E)))
        J_ij[np.ix_(idx_I,idx_I)] = -self.J*self.g*np.random.choice(self.data.sampled_J_II, size=(len(idx_I), len(idx_I)))
        J_ij[np.ix_(idx_I,idx_X)] = self.J  * np.random.choice(self.data.sampled_J_IX, size=np.shape(self.Q[np.ix_(idx_I,idx_X)]))

        return J_ij