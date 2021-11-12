# -*- coding: utf-8 -*-
"""
Assembly detection related utility functions (mostly loading simulation related stuff)
author: András Ecker, last update: 11.2021
"""

import os
import h5py
from collections import namedtuple
import numpy as np
import pandas as pd

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])
SingleCellFeatures = namedtuple("SingleCellFeatures", ["gids", "r_spikes", "mean_ts", "std_ts"])


def _get_bluepy_circuit(circuitconfig_path):
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


def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def get_seeds(root_path):
    """Reads sim seeds from simwriter generated file"""
    f_name = os.path.join(root_path, "project_simulations.txt")
    with open(f_name, "r") as f:
        seeds = [int(line.strip().split('/')[-1][4:]) for line in f]
    return seeds


def get_stim_times(patterns_fname):
    """Return list of stimulus times used during the simulation"""
    with open(patterns_fname, "r") as f:
        stim_times = [float(line.strip().split()[0]) for line in f]
    return stim_times


def get_patterns(patterns_fname):
    """Return list of patterns presented during the simulation"""
    with open(patterns_fname, "r") as f:
        patterns = [line.strip().split()[1] for line in f]
    return patterns


def get_E_gids(c, target):
    from bluepy.enums import Cell
    return c.cells.ids({"$target": target, Cell.SYNAPSE_CLASS: "EXC"})


def _get_layer_E_gids(c, layer, target):
    from bluepy.enums import Cell
    return c.cells.ids({"$target": target, Cell.LAYER: layer, Cell.SYNAPSE_CLASS: "EXC"})


def get_mtypes(c, gids):
    return c.cells.get(gids)["mtype"]


def get_depths(c, gids):
    """Get depths AKA. y-coordinates for v5 circuits"""
    return c.cells.get(gids)["y"]


def get_depths_SSCx(gids):
    """Reads depth values from saved file(s) and return bluepy style Series for SSCx (v7 circuit)"""
    # 1) read András' depths.csv which only has hex_O1
    depths = pd.read_csv("/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/hex_O1_depths.csv", index_col=0)
    if np.isin(gids, depths.index.to_numpy()).all():
        return depths.loc[gids]
    else:  # if that doesn't have all `gids` then read Sirio's depths.txt which covers the whole SSCx
        f_name = "/gpfs/bbp.cscs.ch/data/scratch/proj83/home/bolanos/circuits/Bio_M/20200805/hexgrid/depths.txt"
        data = np.genfromtxt(f_name)
        idx = np.searchsorted(data[:, 0], gids)
        return pd.Series(data[idx, 1], index=gids)


def _guess_circuit_version(target):
    """The version of the circuit: O1.v5 AKA Markram et al. 2015 or SSCx (v7) determines how to get depth values
    as it's simply y coordinate in v5, while it has to be derived from the stream lines of the atlas based SSCx.
    This helper function tries to guess the circuit version from the usual target names we used in those
    circuit versions... which is not ideal. TODO: hardcode in config or come up with a better approach"""
    if "_Column" in target:  # probably v5 circuit
        return "v5"
    elif "hex" in target:  # probably SSCx
        return "v7"
    else:
        raise RuntimeError("Couldn't figure out circuit version from target name: %s" % target)


def get_figure_asthetics(circuit_config, target, gids=None):
    """Gets gid depths, layer boundaries and cell numbers for figure asthetics"""
    c = _get_bluepy_circuit(circuit_config)
    if gids is None:
        gids = get_E_gids(c, target)
    c_version = _guess_circuit_version(target)
    # get depths
    assert c_version in ["v5", "v7"], "Circuit version %s is not supported yet..." % c_version
    if c_version == "v5":
        depths = get_depths(c, gids)
    elif c_version == "v7":
        depths = get_depths_SSCx(gids)
    # get ystuff
    yticks, yticklables, hlines = [], [], []
    for layer in range(2, 7):
        gids = _get_layer_E_gids(c, layer, target)
        yticklables.append("L%i\n(%i)" % (layer, len(gids)))
        ys = depths.loc[gids]
        yticks.append(ys.mean())
        if c_version == "v5":
            if layer == 2:
                hlines.append(ys.max())
                hlines.append(ys.min())
            else:
                hlines.append(ys.min())
        elif c_version == "v7":
            # the SSCx is atlas based and doesn't have clear boundaries, so we'll just use top and bottom
            if layer == 2:
                hlines.append(ys.min())
            elif layer == 6:
                hlines.append(ys.max())
    return depths, {"yticks": yticks, "yticklabels": yticklables, "hlines": hlines}


def get_spikes(sim, gids, t_start, t_end):
    """Extracts spikes (using bluepy)"""
    if gids is None:
        spikes = sim.spikes.get(t_start=t_start, t_end=t_end)
    else:
        spikes = sim.spikes.get(gids, t_start=t_start, t_end=t_end)
    return np.asarray(spikes.index), np.asarray(spikes.values)


def load_assemblies_from_h5(h5f_name, prefix="assemblies", load_metadata=False):
    """Load assemblies over seeds from saved h5 file into dict of AssemblyGroups"""
    from assemblyfire.assemblies import AssemblyGroup, AssemblyProjectMetadata
    with h5py.File(h5f_name, "r") as h5f:
        seeds = list(h5f[prefix].keys())
    assembly_grp_dict = {seed: AssemblyGroup.from_h5(h5f_name, seed, prefix=prefix) for seed in seeds}
    if load_metadata:
        project_metadata = {seed: AssemblyProjectMetadata.from_h5(h5f_name, seed, prefix=prefix) for seed in seeds}
        return assembly_grp_dict, project_metadata
    else:
        return assembly_grp_dict


def load_consensus_assemblies_from_h5(h5f_name, prefix="consensus"):
    """Load consensus (clustered and thresholded )assemblies
    from saved h5 file into dict of ConsensusAssembly objects"""
    from assemblyfire.assemblies import ConsensusAssembly
    with h5py.File(h5f_name, "r") as h5f:
        keys = list(h5f[prefix].keys())
    return {k: ConsensusAssembly.from_h5(h5f_name, k, prefix=prefix) for k in keys}


def load_spikes_from_h5(h5f_name, prefix="spikes"):
    """Load spike matrices over seeds from saved h5 file"""
    from assemblyfire.assemblies import AssemblyProjectMetadata
    h5f = h5py.File(h5f_name, "r")
    seeds = list(h5f[prefix].keys())
    prefix_grp = h5f[prefix]
    spike_matrix_dict = {}
    for seed in seeds:
        spike_matrix_dict[int(seed[4:])] = SpikeMatrixResult(prefix_grp[seed]["spike_matrix"][:],
                                                             prefix_grp[seed]["gids"][:],
                                                             prefix_grp[seed]["t_bins"][:])
    h5f.close()
    project_metadata = AssemblyProjectMetadata.from_h5(h5f_name, prefix=prefix)
    return spike_matrix_dict, project_metadata


def load_single_cell_features_from_h5(h5f_name, prefix="spikes"):
    """Load spike matrices over seeds from saved h5 file"""
    from assemblyfire.assemblies import AssemblyProjectMetadata
    h5f = h5py.File(h5f_name, "r")
    prefix_grp = h5f[prefix]
    single_cell_features = SingleCellFeatures(prefix_grp["gids"][:], prefix_grp["r_spikes"][:],
                                              prefix_grp["mean_ts_in_bin"][:], prefix_grp["std_ts_in_bin"][:])
    h5f.close()
    project_metadata = AssemblyProjectMetadata.from_h5(h5f_name, prefix=prefix)
    return single_cell_features, project_metadata
