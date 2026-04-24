import numpy as np
import bct
from scipy.spatial.distance import pdist, squareform

def identify_ensembles(activity_trace, thr_prop=0.2, gamma=1.0, num_runs=10):
    """
    Args: 
        activity_trace: 形状 (N_neurons, T_steps)
        thr_prop: 比例阈值 (0 < thr_prop < 1)，例如 0.2 表示保留最强的 20% 连接
        gamma: 模块度参数
        num_runs: 运行 Louvain 的次数
    Returns: 
        final_partition: dict mapping n_id to c_id
        sim_matrix: array of shape (N, N)
    """
    N = activity_trace.shape[0]
    
    # 1. 计算余弦相似度矩阵
    # pdist 计算的是 distance，相似度 = 1 - distance
    dist = pdist(activity_trace, metric='cosine')
    sim_matrix = 1 - squareform(dist)
    np.fill_diagonal(sim_matrix, 0)
    sim_matrix[np.isnan(sim_matrix)] = 0
    
    # 2. 比例阈值化与权重归一化 (使用 bctpy)
    CIJ = bct.threshold_proportional(sim_matrix, thr_prop)
    CIJ = bct.weight_conversion(CIJ, 'normalize')
    
    # 3. 多次运行 Louvain 构建共识
    partitions = np.zeros((N, num_runs))
    for i in range(num_runs):
        # bct.community_louvain 默认处理有向/无向加权图
        ci, _ = bct.community_louvain(CIJ, gamma=gamma)
        partitions[:, i] = ci
        
    # 4. 共识聚类
    # 生成一致性矩阵 D 并除以运行次数得到概率
    D = bct.agreement(partitions) / num_runs
    # tau=0.7, reps=200 完全对应 MATLAB: CC = consensus_und(D,0.7,200)
    CC = bct.consensus_und(D, tau=0.7, reps=2)
    
    # 5. 后处理：过滤低模块内连接度和极小社区
    # 计算 module_degree (注意：这里需自行实现类似 MATLAB 中的非 zscore 版本)
    CIJ_bin = (CIJ > 0).astype(float)
    Z = np.zeros(N)
    for c_id in np.unique(CC):
        idx = np.where(CC == c_id)[0]
        if len(idx) > 0:
            # 当前簇内的子图连接度之和
            sub_graph = CIJ_bin[np.ix_(idx, idx)]
            Z[idx] = np.array(sub_graph.sum(axis=1)).flatten()
            
    # 过滤 module degree < 3 的细胞
    CC[Z < 3] = np.nan
    
    # 过滤 size < 5 的社区
    unique_clusters = np.unique(CC[~np.isnan(CC)])
    for c_id in unique_clusters:
        if np.sum(CC == c_id) < 5:
            CC[CC == c_id] = np.nan
            
    # 6. 重新整理 ID
    final_partition = {}
    valid_idx = np.where(~np.isnan(CC))[0]
    
    # 将离散的 ID 映射为连续的 1, 2, 3...
    unique_valid_ids = np.unique(CC[valid_idx])
    id_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_valid_ids)}
    
    print(f"DEBUG: Found {len(unique_valid_ids)} unique clusters (excluding others).")
    for old_id, new_id in id_mapping.items():
        count = int(np.sum(CC == old_id))
        print(f"DEBUG: Cluster {new_id} (old_id {old_id}) has {count} neurons.")

    for n_id in range(N):
        if np.isnan(CC[n_id]):
            final_partition[n_id] = 0  # 0 表示未分类神经元
        else:
            final_partition[n_id] = id_mapping[CC[n_id]]
            
    return final_partition, sim_matrix

def plot_louvain_results(partition, corr_matrix, neuron_coords, save_dir="output/analysis"):
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    
    # ---- 1. Plot Correlation Matrix by Ensembles ----
    clusters = {}
    for n_id, c_id in partition.items():
        clusters.setdefault(c_id, []).append(n_id)
        
    sorted_neurons = []
    ticks = [] 
    tick_labels = []
    current_idx = 0
    unique_clusters = sorted([c for c in clusters.keys() if c != 0])
    lines = []
    for c_id in unique_clusters:
        members = clusters[c_id]
        sorted_neurons.extend(members)
        length = len(members)
        ticks.append(current_idx + length / 2)
        tick_labels.append(str(c_id))
        current_idx += length
        lines.append(current_idx)
        
    if 0 in clusters:
        members = clusters[0]
        sorted_neurons.extend(members)
        length = len(members)
        ticks.append(current_idx + length / 2)
        tick_labels.append("others")
        current_idx += length
        
    sorted_neurons = np.array(sorted_neurons)
    sorted_sim = corr_matrix[np.ix_(sorted_neurons, sorted_neurons)]
    
    plt.figure(figsize=(6, 5))
    vmax = np.percentile(sorted_sim[sorted_sim > 0], 95) if np.sum(sorted_sim > 0) > 0 else 0.5
    im = plt.imshow(sorted_sim, cmap='viridis', aspect='auto', interpolation='none', 
                    vmin=0, vmax=vmax)
    plt.colorbar(im, label='Correlation (r)')
    
    for l in lines:
        plt.axhline(l - 0.5, color='white', linewidth=1)
        plt.axvline(l - 0.5, color='white', linewidth=1)
        
    plt.xticks(ticks, tick_labels, fontsize=12)
    plt.yticks(ticks, tick_labels, fontsize=12)
    plt.xlabel('Neurons of each ensemble', fontsize=14)
    plt.ylabel('Neurons of each ensemble', fontsize=14)
    plt.title('Ensembles', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ensemble_correlation.png"), dpi=300)
    plt.close()

    # ---- 2. Plot Spatial Distribution for Each Ensemble ----
    for c_id in unique_clusters:
        plt.figure(figsize=(4, 4))
        plt.scatter(neuron_coords[:, 0], neuron_coords[:, 1], c='none', edgecolors='lightgrey', s=20, label='GCaMP6s-active neurons')
        
        c_members = clusters[c_id]
        if len(c_members) > 0:
            c_coords = neuron_coords[np.array(c_members)]
            plt.scatter(c_coords[:, 0], c_coords[:, 1], c='black', s=20, label='Ensembles')
            
        plt.text(0.95, 0.05, f"#{c_id}", color='red', fontsize=16, transform=plt.gca().transAxes, ha='right', va='bottom')
        plt.axis('equal')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"ensemble_spatial_{c_id}.png"), dpi=300)
        plt.close()

def plot_ensemble_activity_trace(partition, steady_state_responses, N_theta=8, save_dir="output/analysis"):
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    clusters = {}
    for n_id, c_id in partition.items():
        if c_id != 0: # exclude unclassified
            clusters.setdefault(c_id, []).append(n_id)
            
    unique_clusters = sorted(clusters.keys())
    if len(unique_clusters) == 0:
        print("No valid clusters found for trace plot.")
        return

    num_clusters = len(unique_clusters)
    clusters_to_plot = unique_clusters[:min(num_clusters, 6)]
    num_plots = len(clusters_to_plot)

    colors = ['#8A2BE2', '#DAA520', '#1f77b4', '#e377c2', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(num_plots, 1, figsize=(6, 1.5 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
    
    T_steps = steady_state_responses.shape[2]
    total_frames = N_theta * T_steps
    
    x = np.arange(total_frames)
    
    for i, c_id in enumerate(clusters_to_plot):
        ax = axes[i]
        members = clusters[c_id]
        
        ensemble_trace = steady_state_responses[members, :, :]
        # average across neurons: shape (N_theta, T_steps)
        mean_trace = np.mean(ensemble_trace, axis=0)
        # flatten to string out the stimuli: shape (N_theta * T_steps,)
        flat_trace = mean_trace.flatten()
        
        ax.plot(x, flat_trace, color=colors[i % len(colors)], linewidth=1.5)
        
        # add dashed vertical lines
        for j in range(1, N_theta):
            ax.axvline(j * T_steps, color='k', linestyle='--', alpha=0.5)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-0.02, max(0.8, np.max(flat_trace) * 1.1))
        
        ax.set_yticks([0, 0.4, 0.8])
        ax.text(1.02, 0.5, f"#{c_id}", transform=ax.transAxes, fontsize=14, va='center')
        
        if i == num_plots // 2:
            ax.set_ylabel('Ensembles activity', fontsize=14)
            
    axes[-1].set_xlabel('Frames (sorted)', fontsize=14)
    axes[-1].set_xticks([])

    # Optional: draw orientation icons above the first plot
    # To keep simple, we just leave space or draw simple lines if needed.
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "ensemble_activity_trace.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
