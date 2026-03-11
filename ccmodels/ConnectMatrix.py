import numpy as np

class ConnectMatrix: 
    def __init__(self, k_EE, p_EE, J, g, assign_chi=False, path_2_propdata='../data/sample_data.pkl'):
        self.k_EE = k_EE
        self.p_EE = p_EE
        self.J = J
        self.g = g
        self.assign_chi = assign_chi
        self.path_2_propdata = path_2_propdata

        self.data = np.load(self.path_2_propdata, allow_pickle=True).item()

        # Number of excitatory/inhibitory/tuned neurons
        self.N_E = int(self.k_EE/self.p_EE)
        self.N_I = int(self.data["eta_I"] * self.N_E)
        self.N_X = int(self.data["eta_X"] * self.N_E)

        self.PE = self.N_E / (self.N_E + self.N_I)
        self.PI = self.N_I / (self.N_E + self.N_I)

        # Average connectivity to E neurons per post-synaptic neuron (k_XY means X -> Y)
        self.k_E_TOT = self.k_EE / self.data["gamma_EE"]
        self.k_EE = self.k_EE
        self.k_EI = self.data["gamma_EI"] * self.k_E_TOT
        self.k_EX = self.data["gamma_EX"] * self.k_E_TOT

        # chi = ki/ke
        if assign_chi:
            self.chi = 1.0
        else:
            self.chi = self.data["chi"]

        # Average connectivity to I neurons per post-synaptic neuron
        self.k_I_TOT = self.data["chi"] * self.k_E_TOT
        self.k_IE = self.data["gamma_IE"] * self.k_I_TOT
        self.k_II = self.data["gamma_II"] * self.k_I_TOT
        self.k_IX = self.data["gamma_IX"] * self.k_I_TOT

        # Connection probabilities (p_XY means p(X -> Y))
        self.p_EE = self.k_EE / self.N_E
        self.p_EI = self.k_EI / self.N_I
        self.p_EX = self.k_EX / self.N_X

        self.p_IE = self.k_IE / self.N_I
        self.p_II = self.k_II / self.N_I
        self.p_IX = self.k_IX / self.N_X

        # Number of tuned/untuned neurons
        self.NT_E = int(self.data["etaT_E"] * self.N_E)
        self.NT_X = int(self.data["etaT_X"] * self.N_X)
        self.NU_E = self.N_E - self.NT_E
        self.NU_X = self.N_X - self.NT_X

        # Average connectivity of tuned neurons
        self.kTT_EE = self.data["gammaTT_EE"] * self.k_EE
        self.kTT_EX = self.data["gammaTT_EX"] * self.k_EX
        self.kTU_EE = self.data["gammaTU_EE"] * self.k_EE
        self.kTU_EX = self.data["gammaTU_EX"] * self.k_EX
        self.kUT_EE = self.data["gammaUT_EE"] * self.k_EE
        self.kUT_EX = self.data["gammaUT_EX"] * self.k_EX
        self.kUU_EE = self.data["gammaUU_EE"] * self.k_EE
        self.kUU_EX = self.data["gammaUU_EX"] * self.k_EX

        # Connection probabilities of tuned neurons and untuned neurons
        self.pTT_EE = self.kTT_EE / self.NT_E
        self.pTU_EE = self.kTU_EE / self.NU_E
        self.pUT_EE = self.kUT_EE / self.NT_E
        self.pUU_EE = self.kUU_EE / self.NU_E
        self.pTT_EX = self.kTT_EX / self.NT_X
        self.pTU_EX = self.kTU_EX / self.NU_X
        self.pUT_EX = self.kUT_EX / self.NT_X
        self.pUU_EX = self.kUU_EX / self.NU_X

        # other variables of data
        self.delta_PO = self.data["delta_PO"]
        self.Theta = self.data["Theta"]
        self.ntheta = len(self.Theta)
        self.NhatTheta_E = int(self.NT_E / self.ntheta)
        self.NhatTheta_X = int(self.NT_X / self.ntheta)

        # variables of connections
        self.gammaDTheta_EE = self.data["gammaDTheta_EE"]
        self.gammaDTheta_EX = self.data["gammaDTheta_EX"]

        self.sampled_J_EE = self.data["sampled_J_EE"]
        self.sampled_J_EI = self.data["sampled_J_EI"]
        self.sampled_J_EX = self.data["sampled_J_EX"]
        self.sampled_J_IE = self.data["sampled_J_IE"]
        self.sampled_J_II = self.data["sampled_J_II"]
        self.sampled_J_IX = self.data["sampled_J_IX"]

        self.sampled_ratesT_X = self.data["sampled_ratesT_X"]
        self.sampled_ratesU_X = self.data["sampled_ratesU_X"]

        # result of sample_matrix
        self.Q, self.Assigned_PO = self.sample_matrix()
        # result of sample_J
        self.J_ij = self.sample_J(self.Q, self.J, self.g)
        # result of get_rateX
        self.rate_X_of_Theta = self.get_rateX(self.Assigned_PO)
        # result of compute_orientation_selectivity_index
        self.OSI = self.compute_orientation_selectivity_index(self.rate_X_of_Theta)

    

    def gammaDTheta_EE(self, x):
        # 1. 记录原始形状并拉平
        original_shape = np.shape(x)
        x_flat = np.ravel(x)
        
        # 2. 二分法：寻找插入点 (假设 self.delta_PO 已经是升序的)
        # idx 是第一个大于或等于 x_flat 的元素的索引
        idx = np.searchsorted(self.delta_PO, x_flat, side='left')
        
        # 3. 处理边界：防止 idx-1 或 idx 越界
        # 将索引限制在 [1, len-1]，确保我们总能安全地对比“左邻居”和“当前位”
        idx_clamped = np.clip(idx, 1, len(self.delta_PO) - 1)
        
        # 4. 获取左、右两个候选值的索引
        idx_left = idx_clamped - 1
        idx_right = idx_clamped
        
        val_left = self.delta_PO[idx_left]
        val_right = self.delta_PO[idx_right]
        
        # 5. 比较谁更接近
        # 计算到左边和右边的距离
        dist_left = np.abs(x_flat - val_left)
        dist_right = np.abs(x_flat - val_right)
        
        # 选出距离更小的索引
        closest_indices = np.where(dist_left <= dist_right, idx_left, idx_right)
        
        # 6. 阈值检查 (0.01) 并获取结果
        min_dists = np.where(dist_left <= dist_right, dist_left, dist_right)
        proximity_check = min_dists <= 0.01
        
        # 根据最接近的索引从 gammaDTheta_EE 取值，不满足阈值的赋为 NaN
        result_flat = np.where(proximity_check, self.gammaDTheta_EE[closest_indices], np.nan)
        
        # 7. 还原形状
        return result_flat.reshape(original_shape)

    def gammaDTheta_EX(self, x):
        """
        get the gammaDTheta_EX for the given angle difference x
        parameters:
            x: angle difference. x is expected to be a vector. 
        return:
            result: gammaDTheta_EX (the weight of the connection between the two neurons for the given angle difference x)
        """
        original_shape = np.shape(x)
        x_flat = np.ravel(x)
        idx = np.searchsorted(self.delta_PO, x_flat, side='left')
        idx_clamped = np.clip(idx, 1, len(self.delta_PO) - 1)
        idx_left = idx_clamped - 1
        idx_right = idx_clamped
        val_left = self.delta_PO[idx_left]
        val_right = self.delta_PO[idx_right]
        dist_left = np.abs(x_flat - val_left)
        dist_right = np.abs(x_flat - val_right)
        closest_indices = np.where(dist_left <= dist_right, idx_left, idx_right)
        min_dists = np.where(dist_left <= dist_right, dist_left, dist_right)
        proximity_check = min_dists <= 0.01
        result_flat = np.where(proximity_check, self.gammaDTheta_EX[closest_indices], np.nan)
        return result_flat.reshape(original_shape)

    def sample_matrix(self):
        """
        sample the matrix for the given properties
        parameters:
            None
        return:
            Q: the matrix of the connections
            Assigned_PO: the assigned PO for the neurons
        """
        Q = np.full((self.N_E + self.N_I + self.N_X, self.N_E + self.N_I + self.N_X), np.nan)
        idx_E = np.arange(0, self.N_E)
        idx_I = np.arange(self.N_E, self.N_E + self.N_I)
        idx_X = np.arange(self.N_E + self.N_I, self.N_E + self.N_I + self.N_X)
        Q[np.ix_(idx_E, idx_E)] = np.random.binomial(1, self.p_EE, (self.N_E, self.N_E))
        Q[np.ix_(idx_E, idx_I)] = np.random.binomial(1, self.p_EI, (self.N_E, self.N_I))
        Q[np.ix_(idx_E, idx_X)] = np.random.binomial(1, self.p_EX, (self.N_E, self.N_X))
        Q[np.ix_(idx_I, idx_E)] = np.random.binomial(1, self.p_IE, (self.N_I, self.N_E))
        Q[np.ix_(idx_I, idx_I)] = np.random.binomial(1, self.p_II, (self.N_I, self.N_I))
        Q[np.ix_(idx_I, idx_X)] = np.random.binomial(1, self.p_IX, (self.N_I, self.N_X))
        
        # Tuned-untuned neurons
        idx_T_E = np.arange(0, self.NT_E)
        idx_U_E = np.arange(self.NT_E, self.NT_E + self.NU_E)
        idx_T_X = np.arange(self.N_E + self.N_I, self.N_E + self.N_I + self.NT_X)
        idx_U_X = np.arange(self.N_E + self.N_I + self.NT_X, self.N_E + self.N_I + self.NT_X + self.NU_X)
        Q[np.ix_(idx_T_E, idx_T_E)] = np.random.binomial(1, self.pTT_EE, (self.NT_E, self.NT_E))
        Q[np.ix_(idx_T_E, idx_U_E)] = np.random.binomial(1, self.pTU_EE, (self.NT_E, self.NU_E))
        Q[np.ix_(idx_U_E, idx_T_E)] = np.random.binomial(1, self.pUT_EE, (self.NU_E, self.NT_E))
        Q[np.ix_(idx_U_E, idx_U_E)] = np.random.binomial(1, self.pUU_EE, (self.NU_E, self.NU_E))
        Q[np.ix_(idx_T_E, idx_T_X)] = np.random.binomial(1, self.pTT_EX, (self.NT_E, self.NT_X))
        Q[np.ix_(idx_T_E, idx_U_X)] = np.random.binomial(1, self.pTU_EX, (self.NT_E, self.NU_X))
        Q[np.ix_(idx_U_E, idx_T_X)] = np.random.binomial(1, self.pUT_EX, (self.NU_E, self.NT_X))
        Q[np.ix_(idx_U_E, idx_U_X)] = np.random.binomial(1, self.pUU_EX, (self.NU_E, self.NU_X))


        # Modify connectivity based on tuning properties
        Assigned_PO = np.full(self.N_E + self.N_I + self.N_X, np.nan)
        # 为 E 神经元赋值
        Assigned_PO[:self.ntheta * self.NhatTheta_E].reshape(self.ntheta, -1)[:] = self.Theta[:, None]
        # 为 X 神经元赋值
        start_X = self.N_E + self.N_I
        end_X = start_X + self.ntheta * self.NhatTheta_X
        Assigned_PO[start_X : end_X].reshape(self.ntheta, -1)[:] = self.Theta[:, None]

        # 1. 计算所有配对的角度差矩阵 (ntheta x ntheta)
        # Theta[:, None] 是列向量，Theta[None, :] 是行向量，相减得到差值矩阵
        DPO = self.Theta[:, None] - self.Theta[None, :]

        # 2. 周期性边界处理 (等价于你的 if dPO >= pi ...)
        DPO = (DPO + np.pi) % (2 * np.pi) - np.pi
        # 计算概率矩阵 (ntheta x ntheta)
        P_EE_mat = self.gammaDTheta_EE(DPO) * self.kTT_EE / self.NhatTheta_E
        P_EX_mat = self.gammaDTheta_EX(DPO) * self.kTT_EX / self.NhatTheta_X

        # 1. 将 ntheta x ntheta 的概率矩阵扩展为全尺寸矩阵
        # repeat(..., n, axis=0) 负责复制行，axis=1 负责复制列
        full_P_EE = np.repeat(np.repeat(P_EE_mat, self.NhatTheta_E, axis=0), self.NhatTheta_E, axis=1)
        full_P_EX = np.repeat(np.repeat(P_EX_mat, self.NhatTheta_E, axis=0), self.NhatTheta_X, axis=1)

        # 2. 直接一次性采样，不需要循环
        # E -> E 部分 (假设前 ntheta * NhatTheta_E 是这些被调谐的神经元)
        idx_tuned_E = np.arange(self.ntheta * self.NhatTheta_E)
        Q[np.ix_(idx_tuned_E, idx_tuned_E)] = np.random.binomial(1, full_P_EE)

        # X -> E 部分
        idx_tuned_X = np.arange(self.N_E + self.N_I, self.N_E + self.N_I + self.ntheta * self.NhatTheta_X)
        Q[np.ix_(idx_tuned_E, idx_tuned_X)] = np.random.binomial(1, full_P_EX)

        return Q, Assigned_PO

    def sample_J(self, Q, J, g):
        idx_E = np.arange(0, self.N_E, 1)
        idx_I = np.arange(self.N_E, self.N_E + self.N_I, 1)
        idx_X = np.arange(self.N_E + self.N_I, self.N_E + self.N_I + self.N_X, 1)

        J_ij = np.full_like(Q, np.nan)

        J_ij[np.ix_(idx_E, idx_E)] = J  * np.random.choice(self.sampled_J_EE, size=np.shape(Q[np.ix_(idx_E, idx_E)]))
        J_ij[np.ix_(idx_E, idx_I)] = -J * g * np.random.choice(self.sampled_J_EI, size=np.shape(Q[np.ix_(idx_E, idx_I)]))
        J_ij[np.ix_(idx_E, idx_X)] = J  * np.random.choice(self.sampled_J_EX, size=np.shape(Q[np.ix_(idx_E, idx_X)]))

        J_ij[np.ix_(idx_I, idx_E)] = J  * np.random.choice(self.sampled_J_IE, size=np.shape(Q[np.ix_(idx_I, idx_E)]))
        J_ij[np.ix_(idx_I, idx_I)] = -J * g * np.random.choice(self.sampled_J_II, size=np.shape(Q[np.ix_(idx_I, idx_I)]))
        J_ij[np.ix_(idx_I, idx_X)] = J  * np.random.choice(self.sampled_J_IX, size=np.shape(Q[np.ix_(idx_I, idx_X)]))

        return J_ij


    def get_rateX(self, Assigned_PO):
        rate_X_of_Theta = np.full((self.N_X, self.ntheta), np.nan)
        
        # Tuned neurons
        for idx_Theta in range(self.ntheta):
            idx = np.where(Assigned_PO == self.Theta[idx_Theta])[0]
            idx = np.arange(self.NhatTheta_X * idx_Theta, self.NhatTheta_X * (idx_Theta + 1))
            
            idx_sample = np.random.randint(0, np.shape(self.sampled_ratesT_X)[0], size=len(idx))
            rate_X_of_Theta[idx, :] = np.roll(self.sampled_ratesT_X[idx_sample, :], idx_Theta, axis=1)
            
        start_untuned = self.ntheta * self.NhatTheta_X 
        if start_untuned < self.N_X:
            idx_untuned = np.arange(start_untuned, self.N_X)
            idx_sample = np.random.randint(0, np.shape(self.sampled_ratesU_X)[0], size=len(idx_untuned))
            rate_X_of_Theta[idx_untuned, :] = self.sampled_ratesU_X[idx_sample, :]

        return rate_X_of_Theta

    def compute_orientation_selectivity_index(self, rate_X_of_Theta):
        """
        compute the orientation selectivity index for the given matrix rate_X_of_Theta
        parameters:
            rate_X_of_Theta: the matrix of the rates of the neurons
        return:
            OSI: the orientation selectivity index
        """
        max_rate_X_of_Theta = np.max(rate_X_of_Theta, axis=1)
        idx_max_rate_X_of_Theta = np.argmax(rate_X_of_Theta, axis=1)
        idx_min_rate_X_of_Theta = idx_max_rate_X_of_Theta + 4
        idx_min_rate_X_of_Theta[idx_min_rate_X_of_Theta >= len(self.Theta)] = idx_max_rate_X_of_Theta[idx_min_rate_X_of_Theta >= len(self.Theta)] - 4
        min_rate_X_of_Theta = rate_X_of_Theta[np.arange(rate_X_of_Theta.shape[0]), idx_min_rate_X_of_Theta]

        OSI = (max_rate_X_of_Theta - min_rate_X_of_Theta) / (np.abs(max_rate_X_of_Theta) + np.abs(min_rate_X_of_Theta))
        return OSI