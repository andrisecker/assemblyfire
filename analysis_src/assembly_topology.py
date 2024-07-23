"""
Connectivity analysis (mostly wrappers around `assemblyfire.topology` functions)
last modified: András Ecker 01.2023
"""

import os
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd

from assemblyfire.config import Config
import assemblyfire.utils as utils
import assemblyfire.topology as topology
import assemblyfire.plots as plots

DSET_DEG = "degree"
DSET_CLST = "strength"
DSET_PVALUE = "pvalue"


def _get_spiking_proj_gids(config, sim_config):
    """Loads grouped (to patterns + non-specific) TC gids
    (Could be done easier with adding stuff to the yaml config... but whatever)"""
    tc_spikes = utils.get_grouped_tc_spikes(config.pattern_gids_fname, sim_config, config.t_start, config.t_end)
    _, patterns = utils.get_stimulus_stream(config.input_patterns_fname, config.t_start, config.t_end)
    pattern_names = np.unique(patterns)
    projf_names = list(utils.get_projf_names(sim_config).keys())
    assert len(projf_names) <= 2, "The code assumes max 2 projections, one pattern specific and one non-specific"
    ns_projf_name = np.setdiff1d(projf_names, [config.patterns_projection_name])[0]
    pattern_gids, ns_gids = {}, []
    for name, data in tc_spikes.items():
        if name in pattern_names:
            pattern_gids[name] = np.unique(data["spiking_gids"])
        else:
            ns_gids.extend(np.unique(data["spiking_gids"]))
    all_pattern_gids = []
    for pattern_name, gids in pattern_gids.items():
        all_pattern_gids.extend(gids)
    return {config.patterns_projection_name: np.unique(all_pattern_gids), ns_projf_name: np.unique(ns_gids)}, pattern_gids


def get_proj_innervation(config):
    """Looks up how many projection fibers, and pattern fibers innervate the neurons"""
    from conntility.circuit_models import circuit_connection_matrix

    sim = utils.get_bluepy_simulation(utils.get_sim_path(config.root_path).iloc[0])
    proj_gids, pattern_gids = _get_spiking_proj_gids(config, sim.config)
    c = sim.circuit
    post_gids = utils.get_gids(c, config.target)
    conn_matrices, proj_indegrees, pattern_indegrees = {}, {}, {}
    for proj, pre_gids in proj_gids.items():
        # get (sparse) connectivity matrix between the input fibers and neurons in the circuit
        input_conn_mat = circuit_connection_matrix(c, proj, pre_gids, post_gids).tocsr()
        conn_matrices[proj] = input_conn_mat
        proj_indegrees[proj] = np.array(input_conn_mat.sum(axis=0)).flatten()
        if proj == config.patterns_projection_name:
            # for each pattern get how many pattern fibers innervate the neurons
            for pattern_name, gids in pattern_gids.items():
                pattern_idx = np.in1d(pre_gids, gids, assume_unique=True)
                pattern_indegrees[pattern_name] = np.array(input_conn_mat[pattern_idx].sum(axis=0)).flatten()

    return conn_matrices, proj_indegrees, pattern_indegrees, post_gids


def n_assemblies_from_projs(assembly_grp_dict, proj_indegrees, gids, fig_path, n_bins=21, bin_min_n=10):
    """Plots number of assemblies a neuron participates in vs. indegree from projections"""
    for seed, assembly_grp in assembly_grp_dict.items():
        n_assemblies = np.sum(assembly_grp.as_bool(), axis=1)
        # as assembly groups only store info about spiking gids,
        # we'll need to get rid of the non-spiking ones from the degree vectors...
        spiking_gid_idx = np.in1d(gids, assembly_grp.all, assume_unique=True)
        proj_indegrees_seed = {proj_name: indegrees[spiking_gid_idx] for proj_name, indegrees in proj_indegrees.items()}
        bin_centers_dict, assembly_probs, assembly_probs_low, assembly_probs_high = {}, {}, {}, {}
        for proj_name, indegrees in proj_indegrees_seed.items():
            # same edges as used in `topology.bin_gids_by_innervation()`
            bin_edges = np.hstack(([0], np.linspace(np.percentile(indegrees[indegrees != 0], 1),
                                                    np.percentile(indegrees[indegrees != 0], 99), n_bins)))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_centers_dict[proj_name] = bin_centers
            bin_idx = np.digitize(indegrees, bin_edges, right=True)
            probs = np.zeros_like(bin_centers, dtype=np.float32)
            probs_low, probs_high = np.zeros_like(probs), np.zeros_like(probs)
            for i in range(len(bin_centers)):
                probs[i], probs_low[i], probs_high[i] = topology.prob_with_binom_ci(n_assemblies[bin_idx == i + 1], bin_min_n)
            assembly_probs[proj_name] = probs
            assembly_probs_low[proj_name], assembly_probs_high[proj_name] = probs_low, probs_high
        fig_name = os.path.join(fig_path, "assembly_n_from_projections_seed%s.png" % seed)
        plots.plot_assembly_n_from(bin_centers_dict, assembly_probs, assembly_probs_low, assembly_probs_high,
                                   "In degree from projections", "projections", fig_name)


def assembly_prob_mi_from_proj_ci(assembly_grp_dict, conn_matrices, gids, fig_path,
                                  n_bins=21, bin_min_n=10, sign_th=2):
    """Plots assembly probabilities and (relative) fraction of entropy explained
    from common innervation with projections"""
    proj_names = list(conn_matrices.keys())
    for seed, assembly_grp in assembly_grp_dict.items():
        plot_args = [{proj_name: {} for proj_name in proj_names}, {proj_name: {} for proj_name in proj_names},
                     {proj_name: {} for proj_name in proj_names}, {proj_name: {} for proj_name in proj_names}]
        for proj_name, matrix in conn_matrices.items():
            ci_matrix = matrix.transpose() * matrix
            ci_matrix[range(len(gids)), range(len(gids))] = 0
            proj_cis = {}
            for assembly in assembly_grp.assemblies:
                idx = np.in1d(gids, assembly.gids, assume_unique=True)
                sub_matrix = ci_matrix[:, idx]
                ci = np.array(sub_matrix.sum(axis=1)).flatten().astype(np.float32)
                ci[idx] /= (np.sum(idx) - 1)
                ci[~idx] /= np.sum(idx)
                proj_cis[assembly.idx[0]] = ci

            binned_gids, bin_centers, bin_idx = topology.bin_gids_by_innervation(proj_cis, gids, n_bins)
            # getting MI matrices for all projections:
            mi = topology.assembly_rel_frac_entropy_explained(gids, assembly_grp, bin_centers, bin_idx,
                                                              seed, bin_min_n, sign_th)
            fig_name = os.path.join(fig_path, "frac_entropy_explained_by_%s_CI_%s.png" % (proj_name, seed))
            plots.plot_frac_entropy_explained_by(mi, "%s CI with assembly" % proj_name, fig_name)
            # gathering data for a simplified plot (not the best looking code... but it does the job)
            plot_args_dim = topology.assembly_membership_probability(gids, assembly_grp, binned_gids, bin_centers, bin_min_n)
            for i in range(len(plot_args_dim) - 1):
                for assembly_id in list(plot_args_dim[0].keys()):
                    plot_args[i][proj_name][assembly_id] = plot_args_dim[i][assembly_id][assembly_id]
            if proj_name == proj_names[0]:
                plot_args.append(plot_args_dim[-1])
        # one plot across projections (but only CI with the same assembly)
        fig_name = os.path.join(fig_path, "assembly_prob_from_proj_CI_%s.png" % seed)
        plots.plot_assembly_prob_from(*plot_args, "Common innervation with projections", "projections", fig_name)


def assembly_prob_mi_from_patterns(assembly_grp_dict, pattern_indegrees, gids, fig_path,
                                   n_bins=21, bin_min_n=10, sign_th=2):
    """Plots assembly probabilities and (relative) fraction of entropy explained from pattern indegrees"""
    binned_gids, bin_centers, bin_idx = topology.bin_gids_by_innervation(pattern_indegrees, gids, n_bins)
    for seed, assembly_grp in assembly_grp_dict.items():
        plot_args = topology.assembly_membership_probability(gids, assembly_grp, binned_gids, bin_centers, bin_min_n)
        fig_name = os.path.join(fig_path, "assembly_prob_from_patterns_%s.png" % seed)
        plots.plot_assembly_prob_from(*plot_args, "In degree from patterns", "patterns", fig_name)

        mi = topology.assembly_rel_frac_entropy_explained(gids, assembly_grp, bin_centers, bin_idx,
                                                          seed, bin_min_n, sign_th)
        fig_name = os.path.join(fig_path, "frac_entropy_explained_by_patterns_%s.png" % seed)
        plots.plot_frac_entropy_explained_by(mi, "Innervation by pattern", fig_name)


def assembly_efficacy(config, assembly_grp_dict):
    """Plots synapses initialized at depressed (rho=0) and potentiated (rho=1) states"""
    c = utils.get_bluepy_circuit_from_root_path(config.root_path)
    rhos = utils.get_rho0s(c, config.target)  # get all rhos in one go and then index them as needed

    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Getting efficacies"):
        efficacies = {assembly.idx[0]: rhos.loc[rhos["pre_gid"].isin(assembly.gids)
                                                & rhos["post_gid"].isin(assembly.gids), "rho"].value_counts()
                      for assembly in assembly_grp.assemblies}
        fig_name = os.path.join(config.fig_path, "efficacy_%s.png" % seed)
        plots.plot_efficacy(efficacies, fig_name)


def assembly_in_degrees(assembly_grp_dict, conn_mat, fig_path):
    """Plots indegrees within the assemblies and in their control models"""
    in_degrees, in_degrees_control = topology.in_degree_assemblies(assembly_grp_dict, conn_mat)
    for seed, in_degree in in_degrees.items():
        fig_name = os.path.join(fig_path, "in_degrees_%s.png" % seed)
        plots.plot_in_degrees(in_degree, in_degrees_control[seed], fig_name)
        fig_name = os.path.join(fig_path, "in_degrees_v2_%s.png" % seed)
        plots.plot_in_degree_boxes(in_degree, in_degrees_control[seed], fig_name)


def assembly_simplex_counts(assembly_grp_dict, conn_mat, fig_path):
    """Plots simplex counts in assemblies and control models"""
    simplex_counts, simplex_counts_control = topology.simplex_counts_assemblies(assembly_grp_dict, conn_mat)
    for seed, simplices in simplex_counts.items():
        fig_name = os.path.join(fig_path, "simplex_counts_%s.png" % seed)
        plots.plot_simplex_counts(simplices, simplex_counts_control[seed], fig_name)


def assembly_prob_mi_from_indegree(assembly_grp_dict, conn_mat, fig_path, n_bins=21, bin_min_n=10, sign_th=2):
    """Plots assembly probabilities and (relative) fraction of entropy explained from indegrees"""
    gids = conn_mat.gids
    for seed, assembly_grp in assembly_grp_dict.items():
        assembly_indegrees = {assembly.idx[0]: conn_mat.degree(assembly.gids, gids)
                              for assembly in assembly_grp.assemblies}
        binned_gids, bin_centers, bin_idx = topology.bin_gids_by_innervation(assembly_indegrees, gids, n_bins)

        plot_args = topology.assembly_membership_probability(gids, assembly_grp, binned_gids, bin_centers, bin_min_n)
        palette = {assembly.idx[0]: "pre_assembly_color" for assembly in assembly_grp.assemblies}
        fig_name = os.path.join(fig_path, "assembly_prob_from_indegree_%s.png" % seed)
        plots.plot_assembly_prob_from(*plot_args, "In degree", palette, fig_name)

        mi = topology.assembly_rel_frac_entropy_explained(gids, assembly_grp, bin_centers, bin_idx,
                                                          seed, bin_min_n, sign_th)
        fig_name = os.path.join(fig_path, "frac_entropy_explained_by_recurrent_innervation_%s.png" % seed)
        plots.plot_frac_entropy_explained_by(mi, "Innervation by assembly", fig_name)


def assembly_prob_mi_from_sinks(assembly_grp_dict, conn_mat, palette, fig_path, n_bins=21, bin_min_n=10, sign_th=2):
    """Plots assembly probabilities and (relative) fraction of entropy explained from generalized in degrees
    (sinks of high dim. simplices). Simplices are found in a way that all non-sink neurons
    are guaranteed to be within the assembly"""
    gids = conn_mat.gids
    dims = list(palette.keys())
    for seed, assembly_grp in assembly_grp_dict.items():
        # building simplex sink count lookup
        assembly_simplices = {dim: {} for dim in dims}
        for assembly in tqdm(assembly_grp.assemblies, desc="Getting assembly simplex lists"):
            simplex_list = conn_mat.simplex_list(assembly.gids, gids)
            for dim in dims:
                sink_counts, _ = np.histogram(simplex_list[dim][:, -1], np.arange(len(gids) + 1))
                assembly_simplices[dim][assembly.idx[0]] = sink_counts

        plot_args = [{dim: {} for dim in dims}, {dim: {} for dim in dims}, {dim: {} for dim in dims}, {dim: {} for dim in dims}]
        for dim in dims:
            binned_gids, bin_centers, bin_idx = topology.bin_gids_by_innervation(assembly_simplices[dim], gids, n_bins)
            # getting MI matrices for all dimensions:
            mi = topology.assembly_rel_frac_entropy_explained(gids, assembly_grp, bin_centers, bin_idx,
                                                              seed, bin_min_n, sign_th)
            fig_name = os.path.join(fig_path, "frac_entropy_explained_by_%iDsimplex_sinks_%s.png" % (dim, seed))
            plots.plot_frac_entropy_explained_by(mi, "(Generalized) Innervation by assembly", fig_name)
            # gathering data for a simplified plot (not the best looking code... but it does the job)
            plot_args_dim = topology.assembly_membership_probability(gids, assembly_grp, binned_gids, bin_centers, bin_min_n)
            for i in range(len(plot_args_dim)-1):
                for assembly_id in list(plot_args_dim[0].keys()):
                    plot_args[i][dim][assembly_id] = plot_args_dim[i][assembly_id][assembly_id]
            if dim == dims[0]:
                plot_args.append(plot_args_dim[-1])
        # one plot across dimensions (but only for within assembly simplices)
        fig_name = os.path.join(fig_path, "assembly_prob_from_simplex_dim_%s.png" % seed)
        plots.plot_assembly_prob_from(*plot_args, "Generalized in degree (#simplex sinks)", palette, fig_name, True)


def assembly_prob_mi_from_syn_nnd(assembly_grp_dict, h5f_name, fig_path, n_bins=21, bin_min_n=10, sign_th=2):
    """Plots assembly probabilities and (relative) fraction of entropy explained from
    synapse nearest neighbour distance 'strength' (converts low distances to high 'strength' values)
    (Because these calculations are not parallelized yet it's loading data from HDF5)"""
    with h5py.File(h5f_name, "r") as h5f:
        h5_keys = list(h5f.keys())
    for seed, assembly_grp in assembly_grp_dict.items():
        prefix = "%s_syn_nnd" % seed
        if prefix in h5_keys:  # since this runs forever one might not have the results for all seeds
            syn_nnds = utils.load_syn_nnd_from_h5(h5f_name, len(assembly_grp), prefix=prefix)
            # index out synapse nearest neighbour "strength" from MI DF
            assembly_idx = syn_nnds.columns.get_level_values(0).unique().to_numpy()
            df = syn_nnds.loc[:, (assembly_idx, DSET_CLST)]
            df.columns = [int(assembly_id.split("assembly")[1]) for assembly_id in assembly_idx]
            df = df.loc[:, np.sort(df.columns.to_numpy())]  # order matters for colors...
            gids = df.index.to_numpy()
            # from here it's the same as the other functions with dicts built on the fly
            binned_gids, bin_centers, bin_idx = topology.bin_gids_by_innervation(df, gids, n_bins)

            plot_args = topology.assembly_membership_probability(gids, assembly_grp, binned_gids, bin_centers, bin_min_n)
            palette = {assembly.idx[0]: "pre_assembly_color" for assembly in assembly_grp.assemblies}
            fig_name = os.path.join(fig_path, "assembly_prob_from_syn_nnd_%s.png" % seed)
            plots.plot_assembly_prob_from(*plot_args, "Zscored synapse nnd. strength", palette, fig_name)

            mi = topology.assembly_rel_frac_entropy_explained(gids, assembly_grp, bin_centers, bin_idx,
                                                              seed, bin_min_n, sign_th)
            fig_name = os.path.join(fig_path, "frac_entropy_explained_by_syn_nnd_%s.png" % seed)
            plots.plot_frac_entropy_explained_by(mi, "Synapse nnd. strength from assembly", fig_name)


def assembly_prob_mi_from_indegree_groupedby_syn_nnd(assembly_grp_dict, h5f_name, fig_path,
                                                     n_bins=21, bin_min_n=10, sign_th=2, p_th=0.05):
    """Plots fraction of conditional entropy explained from synapse nearest neighbour distance 'strength'
    (conditioned on indegree) and (within) assembly probabilities from indegrees grouped by
    sign. synapse nearest neighbour distance 'strength' (See doc. of `assembly_prob_mi_from_syn_nnd` above.)"""
    with h5py.File(h5f_name, "r") as h5f:
        h5_keys = list(h5f.keys())
    for seed, assembly_grp in assembly_grp_dict.items():
        prefix = "%s_syn_nnd" % seed
        if prefix in h5_keys:  # since this runs forever one might not have the results for all seeds
            syn_nnds = utils.load_syn_nnd_from_h5(h5f_name, len(assembly_grp), prefix=prefix)
            assembly_idx = syn_nnds.columns.get_level_values(0).unique().to_numpy()
            # quickly check if the syn nnd. strength and indegree are correlated
            fig_name = os.path.join(fig_path, "syn_nnd_indegree_corr_%s.png" % seed)
            plots.plot_joint_dists(syn_nnds.loc[:, (assembly_idx, DSET_CLST)].to_numpy().flatten(),
                                   syn_nnds.loc[:, (assembly_idx, DSET_DEG)].to_numpy().flatten(),
                                   DSET_CLST, DSET_DEG, fig_name)

            # split MI DF to 3 different ones (as that's what the helper functions can deal with...)
            df = syn_nnds.loc[:, (assembly_idx, DSET_PVALUE)]
            df.columns = [int(assembly_id.split("assembly")[1]) for assembly_id in assembly_idx]
            sign = (df < p_th).astype(int)
            df = syn_nnds.loc[:, (assembly_idx, DSET_CLST)]
            df.columns = [int(assembly_id.split("assembly")[1]) for assembly_id in assembly_idx]
            sign[df < 0] *= -1  # `sign` now stores 1 for sign. clustering, -1 for sign. avoidance
            gids = df.index.to_numpy()
            _, _, bin_idx = topology.bin_gids_by_innervation(df, gids, n_bins)
            df = syn_nnds.loc[:, (assembly_idx, DSET_DEG)]
            df.columns = [int(assembly_id.split("assembly")[1]) for assembly_id in assembly_idx]
            df = df.loc[:, np.sort(df.columns.to_numpy())]  # order matters for colors...
            _, bin_centers, bin_idx_cond = topology.bin_gids_by_innervation(df, gids, n_bins)

            sign_keys = {"sing. avoidance": -1, "non-sign.": 0, "sign. 'clustering'": 1}
            plot_args = topology.cond_assembly_membership_probability(gids, assembly_grp, bin_centers, bin_idx_cond,
                                                                      sign, sign_keys, seed, bin_min_n)
            palette = {"sing. avoidance": "gray", "non-sign.": "black", "sign. 'clustering'": "assembly_color"}
            fig_name = os.path.join(fig_path, "assembly_prob_from_indegree_groupedby_syn_nnd_%s.png" % seed)
            plots.plot_assembly_prob_from(*plot_args, "Indegree grouped by synapse nnd.", palette, fig_name)

            mi = topology.assembly_cond_frac_entropy_explained(gids, assembly_grp, bin_idx, bin_idx_cond, seed, sign_th)
            fig_name = os.path.join(fig_path, "cond_frac_entropy_explained_nnd|indegree_%s.png" % seed)
            plots.plot_frac_entropy_explained_by(mi, "Synapse nnd. strength | indegree from assembly", fig_name)


def main(config, assembly_grp_dict, plastic=False):
    fig_path = config.fig_path
    if plastic:
        assembly_efficacy(config, assembly_grp_dict)

    conn_matrices, proj_indegrees, pattern_indegrees, gids = get_proj_innervation(config)
    n_assemblies_from_projs(assembly_grp_dict, proj_indegrees, gids, fig_path)
    assembly_prob_mi_from_proj_ci(assembly_grp_dict, conn_matrices, gids, fig_path)
    assembly_prob_mi_from_patterns(assembly_grp_dict, pattern_indegrees, gids, fig_path)

    conn_mat = topology.AssemblyTopology.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity)
    assembly_in_degrees(assembly_grp_dict, conn_mat, fig_path)
    assembly_simplex_counts(assembly_grp_dict, conn_mat, fig_path)
    assembly_prob_mi_from_indegree(assembly_grp_dict, conn_mat, fig_path)
    assembly_prob_mi_from_sinks(assembly_grp_dict, conn_mat, {2: "gray", 3: "black", 4: "assembly_color"}, fig_path)
    assembly_prob_mi_from_syn_nnd(assembly_grp_dict, config.h5f_name, fig_path)
    assembly_prob_mi_from_indegree_groupedby_syn_nnd(assembly_grp_dict, config.h5f_name, fig_path)


if __name__ == "__main__":
    config = Config("/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/configs/v7_plastic.yaml")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    # assembly_grp = utils.consensus_dict2assembly_grp(utils.load_consensus_assemblies_from_h5(config.h5f_name,
    #                                                        config.h5_prefix_consensus_assemblies))
    # assembly_grp_dict = {"consensus": assembly_grp}

    main(config, assembly_grp_dict)

