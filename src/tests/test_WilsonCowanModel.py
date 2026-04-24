import numpy as np
from src.v1model.default_config import Config
from src.v1model.experimental_data import ExperimentalData
from src.v1model.geometry import L4, L2_3
from src.v1model.SpatialConnectMatrix import SpatialConnectMatrix
from src.v1model.WilsonCowanModel import solve_dynamical_system
from src.v1model.input import L4VisualInput
from src.v1model.NeuronTransfer import tabulate_response

def main():
    cfg = Config()
    exp_data = ExperimentalData(cfg)
    l4 = L4(cfg, exp_data)
    l2_3 = L2_3(cfg, exp_data)
    input = L4VisualInput(l4, cfg)
    scm = SpatialConnectMatrix(l2_3, l4, cfg, exp_data)
    phi_E, phi_I = tabulate_response(cfg)
    aX_func = input.make_aX_func(np.pi/4)
    res = solve_dynamical_system(aX_func, scm.QJ_ij, scm.idx_E, scm.idx_I, scm.idx_X, phi_E, phi_I, cfg)
    print("aE shape: ", res.aE.shape)
    print("aI shape: ", res.aI.shape)
    print("aE_t shape: ", res.aE_t.shape)
    print("aI_t shape: ", res.aI_t.shape)
    print("T_eval shape: ", res.T_eval.shape)
    print("conv_aE: ", res.conv_aE)
    print("conv_aI: ", res.conv_aI)
    
    

if __name__ == "__main__":
    main()