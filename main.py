import numpy as np
import json
from scipy import signal

import typer
app = typer.Typer()

from src.v1model.default_config import Config
from src.v1model.experimental_data import ExperimentalData
from src.v1model.geometry import L2_3

from src.analysis.OSI import get_osi, plot_osi_results
from src.analysis.Louvain import identify_ensembles, plot_louvain_results, plot_ensemble_activity_trace
from src.analysis.distance import plot_spatial_metrics_with_surrogates

@app.command()
def main(
): 
    dir = "output/simulation/260423_223417_L4n40_g1.1"
    with open(f"{dir}/metadata.json") as f:
        metadata = json.load(f)
    
    neuron_coords_all = np.array(metadata["neuron_coords"]) # shape 66*66
    idx_E_all = np.array(metadata["idx_E"])
    L23_distance_all = np.array(metadata["L23_distance"])
    
    # 总体边上有 total_n_side 个，我们要提取中间一半的部分（n_side * n_side个）
    total_n_side = int(np.sqrt(len(neuron_coords_all)))
    start = total_n_side // 4
    n_side = total_n_side // 2
    end = start + n_side
    
    # 找到网格中这部分神经元的原始索引
    grid_indices = np.arange(len(neuron_coords_all)).reshape(total_n_side, total_n_side)
    middle_indices = grid_indices[start:end, start:end].flatten()
    
    # 在中间部分随机选取 1/2 的神经元
    np.random.seed(42)  # 设置随机数种子以保证结果可复现（如果需要每次随机，可以删掉这行）
    sampled_size = len(middle_indices) // 2
    middle_indices = np.random.choice(middle_indices, size=sampled_size, replace=False)
    middle_indices.sort()
    
    # 提取新的 neuron_coords 和 L23_distance
    neuron_coords = neuron_coords_all[middle_indices]
    L23_distance = L23_distance_all[np.ix_(middle_indices, middle_indices)]
    
    # 提取兴奋性神经元中属于中间一半的部分，并映射到新的 idx_E 索引上
    middle_E_mask = np.isin(idx_E_all, middle_indices)
    idx_E_kept = idx_E_all[middle_E_mask]
    
    reverse_map = {old: new for new, old in enumerate(middle_indices)}
    idx_E = np.array([reverse_map[idx] for idx in idx_E_kept])
    
    sim_data_all = np.load(f"{dir}/aE_all.npy") # shape (N_theta, N_E, T_steps)
    sim_data = sim_data_all[:, middle_E_mask, :]
    sim_data_transposed = np.transpose(sim_data, (1, 0, 2)) # shape (N_E, N_theta, T_steps)
    N = sim_data_transposed.shape[0]
    
    T_steps = sim_data_transposed.shape[2]
    steady_state_responses = sim_data_transposed[:, :, int(T_steps*2/3):]
    steady_state_responses_mean = np.mean(steady_state_responses, axis=2)
    activity_trace = steady_state_responses.reshape(N, -1)
    activity_trace_filtered = signal.decimate(activity_trace, int(100/4), axis=-1)
    osi, pref_ori = get_osi(steady_state_responses_mean, np.linspace(0, np.pi, sim_data_transposed.shape[1], endpoint=False))
    partition, corr_matrix = identify_ensembles(activity_trace_filtered)
    
    print(f"shape of neuron_coords: {neuron_coords.shape}")
    print(f"shape of idx_E: {idx_E.shape}")
    print(f"shape of L23_distance: {L23_distance.shape}")
    print(f"shape of sim_data: {sim_data.shape}")
    print(f"shape of osi: {osi.shape}")
    print(f"shape of pref_ori: {pref_ori.shape}")
    print(f"length of partition: {len(partition)}")
    print(f"shape of corr_matrix: {corr_matrix.shape}")

    plot_osi_results(osi, pref_ori, neuron_coords[idx_E])
    plot_louvain_results(partition, corr_matrix, neuron_coords[idx_E])
    plot_ensemble_activity_trace(partition, steady_state_responses, N_theta=sim_data_transposed.shape[1])
    for i in range(4):
        plot_spatial_metrics_with_surrogates(
            partition, 
            L23_distance[np.ix_(idx_E, idx_E)], 
            num_surrogates=100, 
            target_cluster=i,
            save_path=f"output/analysis/spatial_metrics_cluster_{i}.png"
        )
    

if __name__ == "__main__":
    app()