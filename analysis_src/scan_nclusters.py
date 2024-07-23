"""
Scan clustering n and saves plots dendograms, cluster seqs and assemblies for all of them
last modified: András Ecker 02.2023
"""

import os
import warnings
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE

from assemblyfire.config import Config
import assemblyfire.utils as utils
from assemblyfire.clustering import cosine_similarity, detect_assemblies
from assemblyfire.plots import plot_dendogram_silhouettes, plot_tsne, plot_cluster_seqs, plot_distance_corr, plot_db_scores


def get_pattern_distance(locf_name, pklf_name):
    """Gets Earth mover's distance of the input pattern fibers"""
    tmp = np.loadtxt(locf_name)
    gids, pos = tmp[:, 0].astype(int), tmp[:, 1:]
    pattern_gids = utils.get_pattern_gids(pklf_name)
    pattern_pos = {pattern_name: pos[np.in1d(gids, gids_, assume_unique=True), :]
                   for pattern_name, gids_ in pattern_gids.items()}
    pattern_names = np.sort(list(pattern_pos.keys()))
    row_idx, col_idx = np.triu_indices(len(pattern_names), k=1)
    emd = np.zeros_like(row_idx, dtype=np.float32)
    for i, (row_id, col_id) in enumerate(zip(row_idx, col_idx)):
        dists = cdist(pattern_pos[pattern_names[row_id]], pattern_pos[pattern_names[col_id]])
        emd[i] = np.sum(dists[linear_sum_assignment(dists)]) / len(pattern_pos[pattern_names[row_id]])
    return pattern_names, emd


def get_assembly_count_distance(clusters, t_bins, stim_times, patterns, distance_metric="normalized_euclidean"):
    """Gets distance of the assembly counts"""
    _, _, _, pattern_counts = utils.group_clusters_by_patterns(clusters, t_bins, stim_times, patterns)
    # reorder dict to a matrix form
    pattern_names = np.sort(list(pattern_counts.keys()))
    matrix = np.vstack([pattern_counts[pattern_name] for pattern_name in pattern_names])
    if distance_metric == "emd":
        from scipy.stats import wasserstein_distance
        row_idx, col_idx = np.triu_indices(len(pattern_names), 1)
        emd = np.array([wasserstein_distance(matrix[row_id, :], matrix[col_id, :])
                        for row_id, col_id in zip(row_idx, col_idx)])
        return pattern_names, emd
    elif distance_metric == "normalized_euclidean":
        row_idx, col_idx = np.triu_indices(len(pattern_names), 1)
        dists = np.zeros_like(row_idx, dtype=np.float32)
        for i, (row_id, col_id) in enumerate(zip(row_idx, col_idx)):
            a, b = matrix[row_id] - np.mean(matrix[row_id]), matrix[col_id] - np.mean(matrix[col_id])
            dists[i] = np.sqrt(0.5 * np.linalg.norm(a - b) ** 2 / (np.linalg.norm(a) ** 2 + np.linalg.norm(b) ** 2))
        return pattern_names, dists
    else:
        return pattern_names, pdist(matrix, distance_metric)


def cluster_sim_mat(spike_matrix, t_bins, stim_times, patterns, input_pattern_names, input_dist, fig_dir,
                    min_n_clusts=5, max_n_clusts=20):
    """Modified version of `assemblyfire.clustering/cluster_sim_mat()` that plot results for all `n_clusts`
    and some extra stuff"""
    tsne = TSNE(n_components=2, metric="cosine").fit_transform(spike_matrix.T)
    sim_matrix = cosine_similarity(spike_matrix.T)
    dists = 1 - sim_matrix
    dists[dists < 1e-5] = 0.  # fixing numerical errors
    cond_dists = squareform(dists)  # squareform implements its inverse if the input is a square matrix
    linkage_matrix = linkage(cond_dists, method="ward")

    # iterate over number of clusters and plot results for all
    ns, db_scores, n_clusters = np.arange(min_n_clusts, max_n_clusts+1), [], {}
    for n in ns:
        clusters = fcluster(linkage_matrix, n, criterion="maxclust")
        db_scores.append(davies_bouldin_score(dists, clusters))
        clusters -= 1  # this is only needed for assembly detection...
        n_clusters[n] = clusters
        output_pattern_names, output_dist = get_assembly_count_distance(clusters, t_bins, stim_times, patterns)

        plot_dendogram_silhouettes(clusters, linkage_matrix, None, os.path.join(fig_dir, "ward_clustering_n%i.png" % n))
        plot_cluster_seqs(clusters, t_bins, stim_times, patterns, os.path.join(fig_dir, "cluster_seq_n%i.png" % n))
        plot_tsne(clusters, tsne, os.path.join(fig_dir, "t-SNE_n%i.png" % n))
        if (input_pattern_names != output_pattern_names).all():
            warnings.warn("Input and output pattern names don't match!")
        else:
            plot_distance_corr(input_dist, output_dist, os.path.join(fig_dir, "input_vs_output_dists_n%i.png" % n))
    plot_db_scores(ns, db_scores, os.path.join(fig_dir, "db_scores.png"))

    return n_clusters


def main(config_path, seeds, save_assemblies=False):
    config = Config(config_path)
    if seeds == ["_average"]:
        spike_matrix_dict, project_metadata = utils.load_spikes_from_h5(config.h5f_name, config.h5_prefix_avg_spikes)
    else:
        spike_matrix_dict, project_metadata = utils.load_spikes_from_h5(config.h5f_name, config.h5_prefix_spikes)
    stim_times, patterns = project_metadata["stim_times"], project_metadata["patterns"]
    input_pattern_names, input_dist = get_pattern_distance(config.pattern_locs_fname, config.pattern_gids_fname)
    if save_assemblies:
        nrn_loc_df = utils.get_nrn_df(config.h5f_name, config.h5_prefix_connectivity, config.root_path, config.target)

    for seed in seeds:
        spike_matrix, t_bins = spike_matrix_dict["seed%s" % seed].spike_matrix, spike_matrix_dict["seed%s" % seed].t_bins
        fig_dir = os.path.join(config.fig_path, "seed%s_debug" % seed)
        utils.ensure_dir(fig_dir)
        n_clusters = cluster_sim_mat(spike_matrix, t_bins, stim_times, patterns, input_pattern_names, input_dist, fig_dir)
        if save_assemblies:
            # create "fake" dictionary of spike matrices for different ns,
            # so the looping (and proper saving) will be done by `detect_assemblies`
            n_spike_matrix = {n: spike_matrix_dict["seed%s" % seed] for n, _ in n_clusters.items()}
            h5f_name = config.h5f_name.split(".h5")[0] + "_across_ns.h5"
            detect_assemblies(n_spike_matrix, n_clusters, config.core_cell_th_pct, h5f_name,
                              config.h5_prefix_assemblies, nrn_loc_df, fig_dir)


if __name__ == "__main__":
    config_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/configs/v7_np_before.yaml"
    seeds = [1]
    main(config_path, seeds)

