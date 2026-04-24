import numpy as np
import os
from datetime import datetime
import json

import typer
app = typer.Typer()

from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata, interp1d
from src.v1model.default_config import Config
from src.v1model.experimental_data import ExperimentalData
from src.v1model.geometry import L4, L2_3
from src.v1model.SpatialConnectMatrix import SpatialConnectMatrix
from src.v1model.input import L4VisualInput
from src.v1model.NeuronTransfer import tabulate_response
from src.v1model.WilsonCowanModel import WCModel, solve_dynamical_system


@app.command()
def main(
    L4_n_side: int = typer.Option(40, "--L4_n_side", "-l4n", help="number of one side in L4"),
    g: float = typer.Option(1.1, "--g", "-g", help="inhibition-to-excitation ratio"),
):
    np.random.seed(43)
    cfg = Config(L4_n_side=L4_n_side, g=g)
    exp_data = ExperimentalData(cfg)
    l4 = L4(cfg, exp_data)
    l2_3 = L2_3(cfg, exp_data)

    scm = SpatialConnectMatrix(l2_3, l4, cfg, exp_data)
    QJ_ij = scm.QJ_ij
    idx_E = scm.idx_E
    idx_I = scm.idx_I
    idx_X = scm.idx_X

    vis_input = L4VisualInput(l4, cfg)
    phi_E, phi_I = tabulate_response(cfg)

    aE_all = []
    for theta_stim in np.linspace(0, np.pi, cfg.N_theta, endpoint=False):
        aX_func = vis_input.make_aX_func(theta_stim)
        results = solve_dynamical_system(aX_func, QJ_ij, idx_E, idx_I, idx_X, phi_E, phi_I, cfg, T = np.arange(0, 15, 0.1))
        aE_t = results.aE_t # shape (N_E, T_steps)
        aE_all.append(aE_t)
    aE_all = np.array(aE_all) # shape (N_theta, N_E, T_steps)

    metadata = {
        "L4_n_side": L4_n_side,
        "g": g,
        "neuron_coords": l2_3.coords.tolist(),
        "idx_E": idx_E.tolist(),
        "L23_distance": l2_3.get_distance_matrix(cfg.periodic).tolist(),
    }
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    save_path = f"output/simulation/{timestamp}_L4n{L4_n_side}_g{g}/aE_all.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, aE_all)
    with open(f"output/simulation/{timestamp}_L4n{L4_n_side}_g{g}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    


if __name__ == "__main__":
    app()