"""
Assembly detection related utility functions (mostly loading simulation related stuff)
author: András Ecker, last update: 01.2023
"""

import os
import pickle
import h5py
import warnings
from collections import namedtuple
import numpy as np
import pandas as pd
from libsonata import EdgeStorage

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])


def get_bluepy_circuit(circuitconfig_path):
    try:
        from bluepy import Circuit
    except ImportError as e:
        msg = (
            "Assemblyfire requirements are not installed.\n"
            "Please pip install bluepy as follows:\n"
            " pip install -i https://bbpteam.epfl.ch/repository/devpi/simple bluepy[all]"
        )
        raise ImportError(str(e) + "\n\n" + msg)
    return Circuit(circuitconfig_path)


def get_bluepy_simulation(blueconfig_path):
    try:
        from bluepy import Simulation
    except ImportError as e:
        msg = (
            "Assemblyfire requirements are not installed.\n"
            "Please pip install bluepy as follows:\n"
            " pip install -i https://bbpteam.epfl.ch/repository/devpi/simple bluepy[all]"
        )
        raise ImportError(str(e) + "\n\n" + msg)
    return Simulation(blueconfig_path)


def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def get_seeds(root_path):
    """Reads sim seeds from simwriter generated file"""
    f_name = os.path.join(root_path, "project_simulations.txt")
    with open(f_name, "r") as f:
        seeds = [int(line.strip().split('/')[-1][4:]) for line in f]
    return seeds


def get_sim_path(root_path):
    """Loads in simulation paths as pandas (MultiIndex) DataFrame generated by bbp-workflow"""
    pklf_name = os.path.join(root_path, "analyses", "simulations.pkl")
    sim_paths = pd.read_pickle(pklf_name)
    level_names = sim_paths.index.names
    assert len(level_names) == 1 and level_names[0] == "seed", "Only a campaign/DataFrame with single" \
           "`coord`/index level called `seed` is acceptable by assemblyfire"
    return sim_paths


def get_stimulus_stream(f_name, t_start, t_end):
    """Reads the series of presented patterns from .txt file"""
    stim_times, patterns = [], []
    with open(f_name, "r") as f:
        for line in f:
            tmp = line.strip().split()
            stim_times.append(float(tmp[0]))
            patterns.append(tmp[1])
    stim_times, patterns = np.asarray(stim_times), np.asarray(patterns)
    idx = np.where((t_start < stim_times) & (stim_times < t_end))[0]
    return stim_times[idx], patterns[idx]


def get_pattern_gids(pklf_name):
    """Loads VPM gids corresponding to patterns from .pkl file"""
    with open(pklf_name, "rb") as f:
        pattern_gids = pickle.load(f)
    return pattern_gids


def get_gids(c, target):
    return c.cells.ids({"$target": target})


def get_mtype_gids(c, target, mtype):
    from bluepy.enums import Cell
    return c.cells.ids({"$target": target, Cell.MTYPE: mtype})


def get_mtypes(c, gids):
    return c.cells.get(gids)["mtype"]


def get_neuron_locs(circuit_config, target):
    """Gets neuron locations in (supersampled) flat space"""
    from conntility.circuit_models.neuron_groups import load_neurons
    c = get_bluepy_circuit(circuit_config)
    nrn = load_neurons(c, ["layer", "x", "y", "z", "ss_flat_x", "ss_flat_y", "depth"], target)
    return nrn.set_index("gid").drop(columns=["x", "y", "z"])


def get_spikes(sim, gids, t_start, t_end):
    """Extracts spikes (using bluepy)"""
    if gids is None:
        spikes = sim.spikes.get(t_start=t_start, t_end=t_end)
    else:
        spikes = sim.spikes.get(gids, t_start=t_start, t_end=t_end)
    return spikes.index.to_numpy(), spikes.to_numpy()


def get_projf_names(sim_config):
    """Gets the name and edge file path of projections from bluepy.Simulation.config object"""
    projf_names = {}
    projs = sim_config.typed_sections("Projection")
    for proj in projs:
        if hasattr(proj, "Path"):
            projf_names[proj.name] = proj.Path
    return projf_names


def _get_spikef_name(sim_config):
    """Gets the name of the SpikeFile from bluepy.Simulation.config object"""
    f_name = None
    stims = sim_config.typed_sections("Stimulus")
    for stim in stims:
        if hasattr(stim, "SpikeFile"):
            f_name = stim.SpikeFile
            break  # atm. it handles only a single (the first in order) SpikeFile... TODO: extend this
    if f_name is not None:
        f_name = f_name if os.path.isabs(f_name) else os.path.join(sim_config.Run["CurrentDir"], f_name)
    return f_name


def get_tc_spikes(sim_config, t_start, t_end):
    """Loads in input spikes (on projections) using the bluepy.Simulation.config object.
    Returns the format used for plotting rasters and population rates"""
    f_name = _get_spikef_name(sim_config)
    if f_name is not None:
        tmp = np.loadtxt(f_name, skiprows=1)
        spike_times, spiking_gids = tmp[:, 0], tmp[:, 1].astype(int)
        idx = np.where((t_start < spike_times) & (spike_times < t_end))[0]
        return spike_times[idx], spiking_gids[idx]
    else:
        warnings.warn("No SpikeFile found in the BlueConfig, returning empty arrays.")
        return np.array([]), np.array([], dtype=int)


def get_grouped_tc_spikes(pklf_name, sim_config, t_start, t_end):
    """Loads in input spikes (on projections) and groups them by pattern (and POm)"""
    spike_times, spiking_gids = get_tc_spikes(sim_config, t_start, t_end)
    pattern_gids = get_pattern_gids(pklf_name)
    tc_spikes = {pattern_name: {} for pattern_name, _ in pattern_gids.items()}
    tc_spikes["POm"] = {}
    patterns_mask = np.zeros_like(spiking_gids).astype(bool)
    for pattern_name, gids in pattern_gids.items():
        mask = np.in1d(spiking_gids, gids)
        if mask.sum() > 0:
            patterns_mask += mask
            tc_spikes[pattern_name] = {"spike_times": spike_times[mask], "spiking_gids": spiking_gids[mask]}
    pom_mask = ~patterns_mask
    if pom_mask.sum() > 0:
        tc_spikes["POm"] = {"spike_times": spike_times[pom_mask], "spiking_gids": spiking_gids[pom_mask]}
    return tc_spikes


def group_clusters_by_patterns(clusters, t_bins, stim_times, patterns):
    """Groups clustered sign. activity based on the patterns presented"""
    # get basic info (passing them would be difficult...) and initialize empty matrices
    pattern_names, counts = np.unique(patterns, return_counts=True)
    isi, bin_size = np.max(np.diff(stim_times)), np.min(np.diff(t_bins))
    pattern_matrices = {pattern: np.full((np.max(counts), int(isi / bin_size)), np.nan) for pattern in pattern_names}
    # group sign. activity clusters based on patterns
    row_idx = {pattern: 0 for pattern in pattern_names}
    for pattern, t_start, t_end in zip(patterns, stim_times[:-1], stim_times[1:]):
        idx = np.where((t_start <= t_bins) & (t_bins < t_end))[0]
        if len(idx):
            t_idx = (((t_bins[idx] - t_start) / bin_size) - 1).astype(int)
            pattern_matrices[pattern][row_idx[pattern], t_idx] = clusters[idx]
        row_idx[pattern] += 1
    # find max length of sign. activity and cut all matrices there
    max_tidx = np.max([np.sum(~np.all(np.isnan(pattern_matrix), axis=0))
                       for _, pattern_matrix in pattern_matrices.items()])
    pattern_matrices = {pattern_name: pattern_matrix[:, :max_tidx]
                        for pattern_name, pattern_matrix in pattern_matrices.items()}
    # count nr. of clusters per patterns
    pattern_counts, n_clusters = {}, len(np.unique(clusters))
    for pattern_name, matrix in pattern_matrices.items():
        cluster_idx, cluster_counts = np.unique(matrix[~np.isnan(matrix)], return_counts=True)
        counts = np.zeros(n_clusters, dtype=int)
        for i in range(n_clusters):
            if i in cluster_idx:
                counts[i] = cluster_counts[cluster_idx == i]
        pattern_counts[pattern_name] = counts
    return bin_size * max_tidx, row_idx, pattern_matrices, pattern_counts


def count_clusters_by_patterns_across_seeds(all_clusters, t_bins, stim_times, patterns, n_clusters):
    """Counts consensus assemblies across seeds based on the patterns presented"""
    count_matrices = {pattern: np.zeros((len(all_clusters), n_clusters+1), dtype=int) for pattern in np.unique(patterns)}
    seeds = []
    for i, (seed, clusters) in enumerate(all_clusters.items()):
        seeds.append(seed)
        _, _, pattern_matrices, _ = group_clusters_by_patterns(clusters, t_bins[seed], stim_times, patterns)
        for pattern, matrix in pattern_matrices.items():
            cons_assembly_idx, counts = np.unique(matrix, return_counts=True)
            mask = ~np.isnan(cons_assembly_idx)
            for cons_assembly_id, count in zip(cons_assembly_idx[mask], counts[mask]):
                count_matrices[pattern][i, int(cons_assembly_id+1)] = count
    return count_matrices, seeds, np.array([-1] + [i for i in range(n_clusters)])


def load_pkl_df(pklf_name):
    return pd.read_pickle(pklf_name)


def _il_isin(whom, where, parallel):
    """Sirio's in line np.isin() using joblib as parallel backend"""
    if parallel:
        from joblib import Parallel, delayed
        nproc = os.cpu_count() - 1
        with Parallel(n_jobs=nproc, prefer="threads") as p:
            flt = p(delayed(np.isin)(chunk, where) for chunk in np.array_split(whom, nproc))
        return np.concatenate(flt)
    else:
        return np.isin(whom, where)


def get_syn_idx(edgef_name, pre_gids, post_gids, parallel=True):
    """Returns syn IDs between `pre_gids` and `post_gids`
    (~1000x faster than c.connectome.pathway_synapses(pre_gids, post_gids))"""
    edges = EdgeStorage(edgef_name)
    edge_pop = edges.open_population(list(edges.population_names)[0])
    # sonata nodes are 0 based (and the functions expect lists of ints)
    afferents_edges = edge_pop.afferent_edges((post_gids.astype(int) - 1).tolist())
    afferent_nodes = edge_pop.source_nodes(afferents_edges)
    flt = _il_isin(afferent_nodes, pre_gids.astype(int) - 1, parallel=parallel)
    return afferents_edges.flatten()[flt]


def get_syn_properties(c, syn_idx, properties):
    return c.connectome.synapse_properties(syn_idx, properties)


def get_proj_properties(c, proj_name, syn_idx, properties):
    return c.projection(proj_name).synapse_properties(syn_idx, properties)


def get_loc_df(loc_pklf_name, c, target, subtarget):
    """Loads in synapse location related parameters for selected cells
    (and if the target contains other cells than the selected ones get the values for those on the fly."""
    loc_df = load_pkl_df(loc_pklf_name)  # load the saved version
    df_gids = loc_df["post_gid"].unique()
    target_gids = get_gids(c, subtarget)
    # check if there are more gids stored than the target and if so, index out target only
    idx = np.in1d(df_gids, target_gids)
    if idx.sum() < len(idx):
        loc_df = loc_df.loc[loc_df["post_gid"].isin(df_gids[idx])]
    # check if there are any gids missing and if so, get their synapse idx and location related properties
    extra_gids = np.setdiff1d(target_gids, df_gids)
    if len(extra_gids):
        from bluepy.enums import Synapse
        syn_idx = get_syn_idx(c.config["connectome"], get_gids(c, target), extra_gids)
        extra_loc_df = get_syn_properties(c, syn_idx, [Synapse.PRE_GID, Synapse.POST_GID, Synapse.POST_SECTION_ID,
                                                       Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER, Synapse.POST_Z_CENTER])
        extra_loc_df.rename(columns={Synapse.PRE_GID: "pre_gid", Synapse.POST_GID: "post_gid",
                                     Synapse.POST_SECTION_ID: "section_id", Synapse.POST_X_CENTER: "x",
                                     Synapse.POST_Y_CENTER: "y", Synapse.POST_Z_CENTER: "z"}, inplace=True)
        loc_df = pd.concat([loc_df, extra_loc_df]).sort_index()
    return loc_df


def get_rho0s(c, target):
    """Get initial efficacies (rho0_GB in the sonata file) for all EXC synapses in the `target`"""
    from bluepy.enums import Synapse
    gids = get_gids(c, target)
    syn_idx = get_syn_idx(c.config["connectome"], gids, gids)
    syn_df = get_syn_properties(c, syn_idx, [Synapse.PRE_GID, Synapse.POST_GID, "rho0_GB"])
    syn_df.rename(columns={Synapse.PRE_GID: "pre_gid", Synapse.POST_GID: "post_gid", "rho0_GB": "rho"}, inplace=True)
    return syn_df


def save_syn_clusters(save_dir_root, assembly_idx, cluster_df, cross_assembly=True):
    """Saves `cluster_df` with synapse clusters for given assembly"""
    save_dir = os.path.join(save_dir_root, "seed%i" % assembly_idx[1])
    ensure_dir(save_dir)
    if not cross_assembly:
        pklf_name = os.path.join(save_dir, "assembly%i.pkl" % assembly_idx[0])
    else:
        pklf_name = os.path.join(save_dir, "cross_assembly%i.pkl" % (assembly_idx[0]))
    cluster_df.sort_index(inplace=True)
    cluster_df.to_pickle(pklf_name)


def read_base_h5_metadata(h5f_name):
    """Reads ''base'' metadata from h5 attributes (root_path, seeds etc.)"""
    h5f = h5py.File(h5f_name, "r")
    return dict(h5f["spikes"].attrs)


def _read_h5_metadata(h5f, group_name=None, prefix=None):
    """Reads metadata from h5 attributes"""
    if prefix is None:
        prefix = "assemblies"
    prefix_grp = h5f[prefix]
    metadata = dict(prefix_grp.attrs)
    if group_name is not None:
        assert group_name in prefix_grp
        metadata.update(dict(prefix_grp[group_name].attrs))
    return metadata


def load_spikes_from_h5(h5f_name, prefix="spikes"):
    """Load spike matrices over seeds from saved h5 file"""
    h5f = h5py.File(h5f_name, "r")
    seeds = list(h5f[prefix].keys())
    project_metadata = _read_h5_metadata(h5f, prefix=prefix)
    prefix_grp = h5f[prefix]
    spike_matrix_dict = {}
    for seed in seeds:
        spike_matrix_dict[seed] = SpikeMatrixResult(prefix_grp[seed]["spike_matrix"][:],
                                                    prefix_grp[seed]["gids"][:],
                                                    prefix_grp[seed]["t_bins"][:])
    h5f.close()
    return spike_matrix_dict, project_metadata


def load_assemblies_from_h5(h5f_name, prefix="assemblies"):
    """Load assemblies over seeds from saved h5 file into dict of AssemblyGroups"""
    from assemblyfire.assemblies import AssemblyGroup
    h5f = h5py.File(h5f_name, "r")
    seeds = list(h5f[prefix].keys())
    project_metadata = {seed: _read_h5_metadata(h5f, seed, prefix) for seed in seeds}
    h5f.close()
    assembly_grp_dict = {seed: AssemblyGroup.from_h5(h5f_name, seed, prefix=prefix) for seed in seeds}
    return assembly_grp_dict, project_metadata


def load_consensus_assemblies_from_h5(h5f_name, prefix="consensus"):
    """Load consensus (clustered and thresholded )assemblies
    from saved h5 file into dict of ConsensusAssembly objects"""
    from assemblyfire.assemblies import ConsensusAssembly
    with h5py.File(h5f_name, "r") as h5f:
        keys = list(h5f[prefix].keys())
    return {k: ConsensusAssembly.from_h5(h5f_name, k, prefix=prefix) for k in keys}


def consensus_dict2assembly_grp(consensus_assemblies):
    """Create AssemblyGroup (object) from dictionary of consensus assemblies
    (AssemblyGroups are used by several functions investigating connectivity to iterate over assemblies...)"""
    from assemblyfire.assemblies import AssemblyGroup
    cons_assembly_idx = np.sort([int(key.split("cluster")[1]) for key in list(consensus_assemblies.keys())])
    all_gids, assembly_lst = [], []
    for cons_assembly_id in cons_assembly_idx:
        cons_assembly = consensus_assemblies["cluster%i" % cons_assembly_id]
        all_gids.extend(cons_assembly.union.gids)
        cons_assembly.idx = (cons_assembly_id, "consensus")
        assembly_lst.append(cons_assembly)
    return AssemblyGroup(assemblies=assembly_lst, all_gids=np.unique(all_gids), label="ConsensusGroup")


def load_syn_nnd_from_h5(h5f_name, n_assemblies, prefix):
    """Loads synapse nearest neighbour results from h5 file
    pd.read_hdf() doesn't understand the structure, so we need to create an object, and access the DataFrame..."""
    from assemblyfire.syn_nnd import SynNNDResults
    h5f = h5py.File(h5f_name, "r")
    if prefix in list(h5f.keys()):
        results = SynNNDResults(h5f_name, n_assemblies, prefix)
        df = results._df.copy()  # TODO: fix the access in the class
        df.set_index(("gid", "gid"), inplace=True)
        df.index = df.index.astype(int)  # studpid pandas
        df.index.name = "gid"  # studpid pandas
        return df.sort_index()
    else:
        warnings.warn("Prefix: %s not found in HDF5: %s keys. Returning `None`" % (prefix, h5f_name))
        return None


def load_single_cell_features_from_h5(h5f_name, prefix="single_cell"):
    """Load spike matrices over seeds from saved h5 file"""
    h5f = h5py.File(h5f_name, "r")
    prefix_grp = h5f[prefix]
    single_cell_features = {"gids": prefix_grp["gids"][:], "r_spikes": prefix_grp["r_spikes"][:]}
    h5f.close()
    return single_cell_features


def read_cluster_seq_data(h5f_name):
    """Load metadata needed (stored under diff. prefixes) for re-plotting cluster (of time bin) sequences"""
    h5f = h5py.File(h5f_name, "r")
    spikes_metadata = _read_h5_metadata(h5f, prefix="spikes")
    seeds = ["seed%i" % seed for seed in spikes_metadata["seeds"]]
    assemblies_metadata = {seed: _read_h5_metadata(h5f, seed, "assemblies") for seed in seeds}
    metadata = {"clusters": {seed: assemblies_metadata[seed]["clusters"] for seed in seeds},
                "t_bins": {seed: h5f["spikes"][seed]["t_bins"][:] for seed in seeds},
                "stim_times": spikes_metadata["stim_times"],
                "patterns": spikes_metadata["patterns"]}
    h5f.close()
    return metadata
