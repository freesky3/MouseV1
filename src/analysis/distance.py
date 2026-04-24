from scipy.spatial.distance import pdist, squareform
import numpy as np

def analyze_cluster_spatial_metrics(partition, dist_matrix):
    """
    计算每个 ensemble 的 Mean Distance 和 Nearest Neighbor Distance (NND)
    Args: 
        partition: dict mapping n_id to c_id
        dist_matrix: array of shape (N, N)
    """
    cluster_metrics = {}
    
    # 按 cluster 汇总神经元
    clusters = {}
    for n_id, c_id in partition.items():
        clusters.setdefault(c_id, []).append(n_id)
        
    for c_id, members in clusters.items():
        if len(members) < 2: continue
        
        # 提取簇内距离子矩阵
        sub_dist = dist_matrix[np.ix_(members, members)]
        
        # Mean Distance (上三角均值)
        mean_dist = np.mean(sub_dist[np.triu_indices_from(sub_dist, k=1)])
        
        # Nearest Neighbor Distance (掩盖对角线 0 后取每行最小值)
        np.fill_diagonal(sub_dist, np.inf)
        nnd = np.mean(np.min(sub_dist, axis=1))
        
        cluster_metrics[c_id] = {'mean_dist': mean_dist, 'NND': nnd}
        
    return cluster_metrics

import numpy as np
import matplotlib.pyplot as plt

def plot_spatial_metrics_with_surrogates(partition, dist_matrix, num_surrogates=1000, target_cluster=1, save_path="output/analysis/spatial_metrics.png"):
    """
    通过蒙特卡洛（Monte Carlo）随机洗牌策略生成替代数据集（Surrogates），
    并绘制两个相关的子图：
    1. 簇内最近邻距离（NND）的累积概率分布及其 95% 置信区间
    2. 特定簇（Ensemble）的簇内平均距离的累积百分比分布及显著性阈值
    此时传入的 dist_matrix 应为预先计算好的距离矩阵
    """
    
    # 汇总各群组神经元及其大小
    clusters = {}
    for n_id, c_id in partition.items():
        clusters.setdefault(c_id, []).append(n_id)
        
    cluster_sizes = {c_id: len(members) for c_id, members in clusters.items() if len(members) >= 2}
    all_neurons = np.arange(dist_matrix.shape[0])
    
    # ==========================
    # 1. 计算真实情况的度量
    # ==========================
    actual_NNDs = []
    actual_mean_dist = {}
    
    for c_id, members in clusters.items():
        if len(members) < 2: continue
        sub_dist = dist_matrix[np.ix_(members, members)]
        
        # Mean Distance
        actual_mean_dist[c_id] = np.mean(sub_dist[np.triu_indices_from(sub_dist, k=1)])
        
        # NND (对每一个簇内神经元，找簇内离它最近的距离)
        np.fill_diagonal(sub_dist, np.inf)
        actual_NNDs.extend(np.min(sub_dist, axis=1))
        
    actual_NNDs = np.sort(actual_NNDs)
    
    # ==========================
    # 2. 蒙特卡洛随机洗牌
    # ==========================
    surrogate_NNDs_matrix = [] # 保存每一次 surrogate 下所有神经元的 NND，必须确保每一次都有相同的神经元数量
    target_cluster_random_mean_dists = []
    
    for _ in range(num_surrogates):
        shuffled_neurons = np.random.permutation(all_neurons)
        
        idx = 0
        surr_NNDs_this_round = []
        for c_id, size in cluster_sizes.items():
            members = shuffled_neurons[idx:idx+size]
            idx += size
            
            sub_dist = dist_matrix[np.ix_(members, members)]
            
            # 记录 target_cluster 的 random mean distance
            if c_id == target_cluster:
                target_cluster_random_mean_dists.append(
                    np.mean(sub_dist[np.triu_indices_from(sub_dist, k=1)])
                )
                
            np.fill_diagonal(sub_dist, np.inf)
            surr_NNDs_this_round.extend(np.min(sub_dist, axis=1))
            
        surrogate_NNDs_matrix.append(np.sort(surr_NNDs_this_round))
        
    surrogate_NNDs_matrix = np.array(surrogate_NNDs_matrix) # shape: (num_surrogates, num_valid_neurons)
    
    # 计算均值和置信区间（沿着样本维度求百分位数）
    mean_surrogate_NND = np.mean(surrogate_NNDs_matrix, axis=0)
    percentile_2_5 = np.percentile(surrogate_NNDs_matrix, 2.5, axis=0)
    percentile_97_5 = np.percentile(surrogate_NNDs_matrix, 97.5, axis=0)
    
    # ==========================
    # 3. 绘图 (分成两个单独的图)
    # ==========================
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # ---- 面板 F: NND 的累积概率分布 ----
    plt.figure(figsize=(5, 4.5))
    y_vals = np.linspace(0, 1, len(actual_NNDs))
    
    plt.plot(actual_NNDs, y_vals, color='black', linewidth=2, label='Ensembles')
    plt.plot(mean_surrogate_NND, y_vals, color='darkgray', linewidth=1.5, label='Surrogates')
    plt.plot(percentile_2_5, y_vals, color='gray', linestyle='--', linewidth=1, label='95% confidence')
    plt.plot(percentile_97_5, y_vals, color='gray', linestyle='--', linewidth=1)
    
    plt.xlabel('Nearest neighbor\ndistance (NND, μm)', fontsize=12)
    plt.ylabel('Cumulative probability', fontsize=12)
    plt.ylim(-0.02, 1.02)
    plt.legend(frameon=False, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_NND.png'), dpi=300)
    plt.close()
    
    # ---- 面板 G: 特定簇的随机平均距离分布 ----
    plt.figure(figsize=(5, 4.5))
    if target_cluster in cluster_sizes:
        sorted_rand_md = np.sort(target_cluster_random_mean_dists)
        y_rand = np.linspace(0, 100, len(sorted_rand_md))
        
        actual_md_val = actual_mean_dist[target_cluster]
        # 常用的单侧 5% 显著性阈值
        threshold_val = np.percentile(sorted_rand_md, 5) 
        
        plt.plot(sorted_rand_md, y_rand, color='darkgray', linewidth=3, label='Random')
        plt.axvline(x=actual_md_val, color='coral', linestyle='--', linewidth=2, label='Ensemble')
        plt.axvline(x=threshold_val, color='cornflowerblue', linestyle='--', linewidth=2, label='Threshold')
        
        plt.xlabel('Mean distance (μm)', fontsize=12)
        plt.ylabel('Cum. % of mean distance', fontsize=12)
        plt.title(f'#{target_cluster} ensemble', fontsize=14)
        plt.legend(frameon=False, loc='lower right')
        plt.ylim(-2, 102)
    else:
        plt.text(0.5, 0.5, f"Cluster #{target_cluster} not found\nor has <2 members.", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path.replace('.png', f'_MeanDist.png'), dpi=300)
    plt.close()