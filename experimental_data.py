import numpy as np

class ExperimentalData: 
    def __init__(self, N_X, p_EE, assign_chi=False, path_2_propdata='data/sample_data.pkl'):
        self.p_EE = p_EE
        self.assign_chi = assign_chi # False
        self.path_2_propdata = path_2_propdata # data/sample_data.pkl

        self.data = np.load(self.path_2_propdata, allow_pickle=True)

        self.eta_I = self.data["eta_I"] # 0.1805
        self.eta_X = self.data["eta_X"] # 0.4382

        # Number of excitatory/inhibitory/tuned neurons
        # self.N_E = int(self.k_EE/self.p_EE)
        # self.N_I = int(self.data["eta_I"] * self.N_E)
        # self.N_X = int(self.data["eta_X"] * self.N_E)
        self.N_X = N_X
        exact_N_E = N_X / self.eta_X
        exact_N_I = exact_N_E * self.eta_I
        self.L2_3_n_side = int(np.ceil(np.sqrt(exact_N_E + exact_N_I)))
        N_E_N_I = self.L2_3_n_side**2
        self.N_E = int(N_E_N_I / (1+self.eta_I))
        self.N_I = N_E_N_I - self.N_E

        self.PE = self.N_E / (self.N_E + self.N_I) # 0.8475
        self.PI = self.N_I / (self.N_E + self.N_I) # 0.1525

        # Average connectivity to E neurons per post-synaptic neuron (k_XY means X -> Y)
        self.k_EE = self.p_EE * self.N_E # 5
        self.k_E_TOT = self.k_EE / self.data["gamma_EE"]
        self.k_EI = self.data["gamma_EI"] * self.k_E_TOT
        self.k_EX = self.data["gamma_EX"] * self.k_E_TOT

        # chi = ki/ke
        if assign_chi:
            self.chi = 1.0 # 1.0736
        else:
            self.chi = self.data["chi"] # 1.0736

        # Average connectivity to I neurons per post-synaptic neuron
        self.k_I_TOT = self.data["chi"] * self.k_E_TOT
        self.k_IE = self.data["gamma_IE"] * self.k_I_TOT
        self.k_II = self.data["gamma_II"] * self.k_I_TOT
        self.k_IX = self.data["gamma_IX"] * self.k_I_TOT

        # Connection probabilities (p_XY means p(X -> Y))
        # self.p_EE = p_EE
        self.p_EI = self.k_EI / self.N_I # 0.4060
        self.p_EX = self.k_EX / self.N_X # 0.0409

        self.p_IE = self.k_IE / self.N_I # 0.6624
        self.p_II = self.k_II / self.N_I # 0.3526
        self.p_IX = self.k_IX / self.N_X # 0.0514

        # Number of tuned/untuned neurons
        self.NT_E = int(self.data["etaT_E"] * self.N_E)
        self.NT_X = int(self.data["etaT_X"] * self.N_X)
        self.NU_E = self.N_E - self.NT_E
        self.NU_X = self.N_X - self.NT_X

        # possibility of tuned neurons
        self.pT_E = self.NT_E / self.N_E # 0.5600
        self.pT_X = self.NT_X / self.N_X # 0.5714
        self.pU_E = self.NU_E / self.N_E # 0.4400
        self.pU_X = self.NU_X / self.N_X # 0.4286

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
        self.pTT_EE = self.kTT_EE / self.NT_E # 0.1000
        self.pTU_EE = self.kTU_EE / self.NU_E # 0.1000
        self.pUT_EE = self.kUT_EE / self.NT_E # 0.0998
        self.pUU_EE = self.kUU_EE / self.NU_E # 0.1002
        self.pTT_EX = self.kTT_EX / self.NT_X # 0.0427
        self.pTU_EX = self.kTU_EX / self.NU_X # 0.0385
        self.pUT_EX = self.kUT_EX / self.NT_X # 0.0401
        self.pUU_EX = self.kUU_EX / self.NU_X # 0.0419

        # other variables of data
        self.delta_PO = self.data["delta_PO"] # [-3.1416, -2.7489, -2.3562, -1.9635, -1.5708, -1.1781, -0.7854, -0.3927, 0. , 0.3927, 0.7854, 1.1781, 1.5708, 1.9635, 2.3562, 2.7489]
        self.Theta = self.data["Theta"] # Theta: [0.     0.3927 0.7854 1.1781 1.5708 1.9635 2.3562 2.7489 3.1416 3.5343 3.927  4.3197 4.7124 5.1051 5.4978 5.8905]
        self.ntheta = len(self.Theta) # 16
        self.NhatTheta_E = int(self.NT_E / self.ntheta)
        self.NhatTheta_X = int(self.NT_X / self.ntheta)

        # variables of connections
        self.gammaDTheta_EE = self.data["gammaDTheta_EE"] # [0.0742, 0.0665, 0.0574, 0.0551, 0.0549, 0.0551, 0.0578, 0.0725, 0.0872, 0.0725, 0.0578, 0.0551, 0.0549, 0.0551, 0.0574, 0.0665]
        self.gammaDTheta_EX = self.data["gammaDTheta_EX"] # [0.0659, 0.0637, 0.0604, 0.059 , 0.0587, 0.0587, 0.0593, 0.0674, 0.0797, 0.0674, 0.0593, 0.0587, 0.0587, 0.059 , 0.0604, 0.0637]

        self.sampled_J_EE = self.data["sampled_J_EE"]
        self.sampled_J_EI = self.data["sampled_J_EI"]
        self.sampled_J_EX = self.data["sampled_J_EX"]
        self.sampled_J_IE = self.data["sampled_J_IE"]
        self.sampled_J_II = self.data["sampled_J_II"]
        self.sampled_J_IX = self.data["sampled_J_IX"]

        self.sampled_ratesT_X = self.data["sampled_ratesT_X"]
        self.sampled_ratesU_X = self.data["sampled_ratesU_X"]

if __name__ == "__main__":
    exp_data = ExperimentalData(N_X=400, p_EE=0.1)
    for key in exp_data.__dict__:
        if key == "data":
            continue
        elif isinstance(exp_data.__dict__[key], np.ndarray):
            print(key, ": ", exp_data.__dict__[key].shape)
        else:
            print(key, ": ", exp_data.__dict__[key])