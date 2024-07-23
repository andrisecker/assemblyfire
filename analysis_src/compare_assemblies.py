"""
Load assemblies from (either from 2 HDF5 files, or from different prefixes) and compare them
last modified: András Ecker 04.2023
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

import assemblyfire.utils as utils
from assemblyfire.config import Config
from assemblyfire.clustering import pairwise_correlation_x
from assemblyfire.plots import plot_assembly_similarities, plot_pw_corrs_pairs,\
                               plot_consensus_vs_average_assembly_composition


def get_assembly_similarities(assembly_grp1, assembly_grp2, restrict2gids=None):
    """Gets Jaccard distance of assembly neurons"""
    # creating ID matrices with the union of the gids
    gids1, gids2 = assembly_grp1.all, assembly_grp2.all
    gids = np.union1d(gids1, gids2)
    if restrict2gids is not None:
        gids = gids[np.in1d(gids, restrict2gids, assume_unique=True)]
        assembly_idx1 = np.zeros((len(assembly_grp1), len(gids)), dtype=int)
        for assembly in assembly_grp1.assemblies:
            assembly_idx1[assembly.idx[0], np.in1d(gids, assembly.gids, assume_unique=True)] = 1
        assembly_idx2 = np.zeros((len(assembly_grp2), len(gids)), dtype=int)
        for assembly in assembly_grp2.assemblies:
            assembly_idx2[assembly.idx[0], np.in1d(gids, assembly.gids, assume_unique=True)] = 1
    else:
        assembly_idx1 = np.zeros((len(assembly_grp1), len(gids)), dtype=int)
        assembly_idx1[:, np.in1d(gids, gids1, assume_unique=True)] = assembly_grp1.as_bool().transpose().astype(int)
        assembly_idx2 = np.zeros((len(assembly_grp2), len(gids)), dtype=int)
        assembly_idx2[:, np.in1d(gids, gids2, assume_unique=True)] = assembly_grp2.as_bool().transpose().astype(int)
    return 1 - cdist(assembly_idx1, assembly_idx2, "jaccard")


def _get_label(h5f_name):
    return os.path.split(h5f_name)[1].split('.')[0]


def assembly_similarities_from2configs(config1_path, config2_path, restrict=True):
    """Loads in assemblies and gets their Jaccard similarity (seed by seed)"""
    config1 = Config(config1_path)
    config2 = Config(config2_path)
    xlabel, ylabel = _get_label(config2.h5f_name), _get_label(config1.h5f_name)
    assembly_grp_dict1, _ = utils.load_assemblies_from_h5(config1.h5f_name, config1.h5_prefix_assemblies)
    assembly_grp_dict2, _ = utils.load_assemblies_from_h5(config2.h5f_name, config2.h5_prefix_assemblies)
    for seed, assembly_grp1 in assembly_grp_dict1.items():
        restrict2gids = utils.get_gids(utils.get_bluepy_circuit_from_root_path(config2.root_path), config2.target)\
            if restrict else None
        similarities = get_assembly_similarities(assembly_grp1, assembly_grp_dict2[seed], restrict2gids)
        fig_name = os.path.join(config2.fig_path, "assembly_similarities_%s.png" % seed)
        plot_assembly_similarities(similarities, xlabel, ylabel, fig_name)


def consensus_at_fraction_thresholds(consensus_assembly_grp, frac_ths):
    """Create assembly groups (as those are used in this script) at different thresholds, where the thresholded variable
    is number of times contained AKA. `raw` method in `assemblyfire.assemblies()` over the number of instantiations
    TODO: maybe add this method to the `ConsensusAssembly` class"""
    from assemblyfire.assemblies import Assembly, AssemblyGroup
    all_gids = consensus_assembly_grp.all
    assembly_grps = {}
    for frac_th in frac_ths:
        assembly_lst = []
        for assembly in consensus_assembly_grp.assemblies:
            # `fracs` could be stored instead of recalculated again for all thresholds but whatever...
            fracs = assembly.__number_of_times_contained__() / len(assembly.instantiations)
            assembly_lst.append(Assembly(assembly.union.gids[fracs > frac_th], index=(assembly.idx[0], frac_th)))
        assembly_grps[frac_th] = AssemblyGroup(assembly_lst, all_gids)
    return assembly_grps


def consensus_vs_average_assembly_similarity(config_path, frac_ths):
    """Loads in consensus and average assemblies and gets their Jaccard similarity"""
    config = Config(config_path)
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_avg_assemblies)
    avg_assembly_grp = assembly_grp_dict["seed_average"]
    consensus_assemblies = utils.load_consensus_assemblies_from_h5(config.h5f_name, config.h5_prefix_consensus_assemblies)
    consensus_assembly_grp = utils.consensus_dict2assembly_grp(consensus_assemblies)

    # core (as defined bassed on p-value threshold vs. average assembly)
    similarities = get_assembly_similarities(avg_assembly_grp, consensus_assembly_grp)
    fig_name = os.path.join(config.fig_path, "consensus_vs_average_assembly_similarities.png")
    plot_assembly_similarities(similarities, "Consensus (core) assembly", "Average assembly", fig_name)

    # different thresholds, but not on p-value, but on franction of inst. contained instead
    assembly_grps = consensus_at_fraction_thresholds(consensus_assembly_grp, frac_ths)
    for frac_th, assembly_grp in assembly_grps.items():
        similarities = get_assembly_similarities(avg_assembly_grp, assembly_grp)
        fig_name = os.path.join(config.fig_path, "consensus_@frac%.1f_vs_average_assembly_similarities.png" % frac_th)
        plot_assembly_similarities(similarities, "Consensus @frac%.1f assembly" % frac_th, "Average assembly", fig_name)


def consensus_vs_average_assembly_composition(config_path, avg_assembly_id, consensus_assembly_id):
    """TODO"""
    config = Config(config_path)
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_avg_assemblies)
    avg_assembly_grp = assembly_grp_dict["seed_average"]
    avg_gids = avg_assembly_grp.loc(avg_assembly_id).gids
    consensus_assemblies = utils.load_consensus_assemblies_from_h5(config.h5f_name, config.h5_prefix_consensus_assemblies)
    consensus_assembly = consensus_assemblies["cluster%i" % consensus_assembly_id]

    gids, ns = consensus_assembly.union.gids, consensus_assembly.__number_of_times_contained__()
    gids_at_n = [gids[ns == n] for n in np.unique(ns)]
    intersection_at_n = [len(np.intersect1d(gids, avg_gids, assume_unique=True)) for gids in gids_at_n]
    diff_at_n = [len(np.setdiff1d(gids, avg_gids, assume_unique=True)) for gids in gids_at_n]
    fig_name = os.path.join(config.fig_path, "consenses%i_vs_average%i.png" % (consensus_assembly_id, avg_assembly_id))
    plot_consensus_vs_average_assembly_composition(intersection_at_n, diff_at_n, fig_name)


def _load_avg_spikes(config):
    """Load average assembly"""
    spike_matrix_dict, _ = utils.load_spikes_from_h5(config.h5f_name, config.h5_prefix_avg_spikes)
    return spike_matrix_dict["seed_average"].spike_matrix, spike_matrix_dict["seed_average"].gids


def group_gids(assembly_grp1, assembly_idx1, assembly_grp2, assembly_idx2):
    """Group gids (based on assembly membership) for plotting"""
    assembly_gids1 = [assembly_grp1.loc((assembly_id, "consensus")).gids for assembly_id in assembly_idx1]
    assembly_gids2 = [assembly_grp2.loc((assembly_id, "consensus")).gids for assembly_id in assembly_idx2]
    if len(assembly_gids1) == 1 and len(assembly_gids2) == 1:  # same assembly, different neurons
        pass  # TODO
    if len(assembly_gids2) > 1:  # split
        # first intersect them with assembly1
        assembly_gids2 = [np.intersect1d(assembly_gids1[0], assembly_gids2[i], assume_unique=True)
                          for i in range(len(assembly_gids2))]
        assert len(assembly_gids2) == 2, "atm. the ordering is based on 2 resulting assemblies (due to the intersection)"
        assembly2_inters = np.intersect1d(assembly_gids2[0], assembly_gids2[1], assume_unique=True)
        gids = [np.setdiff1d(assembly_gids2[0], assembly2_inters, assume_unique=True), assembly2_inters,
                np.setdiff1d(assembly_gids2[1], assembly2_inters, assume_unique=True)]
        gids.append(np.setdiff1d(assembly_gids1[0], np.concatenate(gids), assume_unique=True))
        idx_ranges = [[0, len(assembly_gids2[0])], [len(gids[0]), len(gids[0]) + len(assembly_gids2[1])]]
        xticks = [int(np.mean(idx_ranges[0])), int(np.mean(idx_ranges[1]))]
        gids = np.concatenate(gids)
        plotting = [xticks, [int(len(gids) / 2)], assembly_idx2.tolist(), assembly_idx1.tolist()]
        return gids, idx_ranges, plotting
    if len(assembly_gids1) > 1:  # merge
        pass  # TODO


def corrs2df(corrs, assembly_idx, idx_ranges):
    """Splits correlation matrix to submatrices (defining assemblies and non-assemblies)
    and puts values into a DataFrame (for easier plotting stat. testing afterwards)"""
    assert (len(assembly_idx) == len(idx_ranges))
    mask = np.ones_like(corrs, dtype=bool)
    sub_corrs, locs = [], []
    for assembly_id, idx_range in zip(assembly_idx, idx_ranges):
        tmp = np.asarray(corrs[idx_range[0]:idx_range[1], idx_range[0]:idx_range[1]]).flatten()
        sub_corrs.append(tmp)
        locs.append(np.full(tmp.shape, "block-diag%i" % assembly_id, dtype="object"))
        mask[idx_range[0]:idx_range[1], idx_range[0]:idx_range[1]] = False
    sub_corrs.append(np.asarray(corrs[mask]).flatten())
    locs.append(np.full(mask.sum(), "off-diag", dtype="object"))
    df = pd.DataFrame(data=np.concatenate(sub_corrs), columns=["corr"])
    df["loc"] = np.concatenate(locs)
    return df


def analyze_corrs(config1_path, config2_path, xlabel=None, ylabel=None, sim_th=0.3):
    """Analyse correlations of assembly pairs (after getting their similarities)"""
    config1 = Config(config1_path)
    config2 = Config(config2_path)
    if xlabel is None and ylabel is None:  # TODO: handle separate cases
        xlabel, ylabel = _get_label(config2.h5f_name), _get_label(config1.h5f_name)
    # load consensus assemblies and average spikes
    assembly_grp1 = utils.consensus_dict2assembly_grp(utils.load_consensus_assemblies_from_h5(config1.h5f_name,
                                                            config1.h5_prefix_consensus_assemblies))
    assembly_grp2 = utils.consensus_dict2assembly_grp(utils.load_consensus_assemblies_from_h5(config2.h5f_name,
                                                            config2.h5_prefix_consensus_assemblies))
    spike_matrix1, spiking_gids1 = _load_avg_spikes(config1)
    spike_matrix2, spiking_gids2 = _load_avg_spikes(config2)
    # get similarities
    similarities = get_assembly_similarities(assembly_grp1, assembly_grp2)
    fig_name = os.path.join(config2.fig_path, "consensus_assembly_similarities.png")
    plot_assembly_similarities(similarities, "assemblies %s " % xlabel, "assemblies %s " % ylabel, fig_name)
    # threshold similarities and detect splitting and merging
    idx1, idx2 = np.where(similarities > sim_th)
    vals1, counts1 = np.unique(idx1, return_counts=True)
    vals2, counts2 = np.unique(idx2, return_counts=True)

    if np.any(counts1 > 1):  # splitting
        split_assembly_idx = vals1[counts1 > 1]
        for assembly_id in split_assembly_idx:
            assembly_idx = idx2[idx1 == assembly_id]
            gids, idx_ranges, plotting = group_gids(assembly_grp1, np.array([assembly_id]), assembly_grp2, assembly_idx)
            corrs1 = pairwise_correlation_x(csr_matrix(spike_matrix1[np.in1d(spiking_gids1, gids), :], dtype=np.float32))
            corrs2 = pairwise_correlation_x(csr_matrix(spike_matrix2[np.in1d(spiking_gids2, gids), :], dtype=np.float32))
            df = corrs2df(corrs2 - corrs1, assembly_idx, idx_ranges)
            fig_name = os.path.join(config2.fig_path, "consensus_assembly%i_split.png" % assembly_id)
            plot_pw_corrs_pairs(corrs1.copy(), corrs2.copy(), df, xlabel, ylabel, *plotting, fig_name, vlines=idx_ranges)
    # if np.any(counts2 > 1):  # merging
    #     pass


if __name__ == "__main__":
    # config1_path = "../configs/v7_5seeds_np_before.yaml"
    # config2_path = "../configs/v7_5seeds_np_after.yaml"
    # analyze_corrs(config1_path, config2_path, ylabel="before", xlabel="after")
    # config_path = "../configs/v7_10seeds_np.yaml"
    # consensus_vs_average_assembly_similarity(config_path, frac_ths=[0.2, 0.4, 0.6, 0.8])
    # consensus_vs_average_assembly_composition(config_path, 7, 1)
    assembly_similarities_from2configs("../configs/v7_np_before.yaml", "../configs/v7_np_after.yaml")

