"""
Rerun single cells in BGLibPy (for extra reporting and access to parameters to be modified)
authors: András Ecker, Sirio Bolaños Puchet; last update: 01.2023
"""

import os
import gc
import time
import logging
import numpy as np
import pandas as pd

import bglibpy

from assemblyfire.config import Config
import assemblyfire.utils as utils

L = logging.getLogger("assemblyfire")


def get_gid_instantiation_vars(ssim):
    """Gets variables necessary to (properly) instantiate a gid in BGLibPy"""
    ca = ssim.circuit_access
    # on has to manually add TC spikes (as BGLibPy doesn't load spikes from the SpikeFile in the BlueConfig)
    proj_spike_trains = utils.get_tc_spikes_bglibpy(ca.config.bc)
    # instead of all the cell's synapses get only the ones that originate from the sim's target
    # and the ones from the active TC fibers (TC fibers don't have minis - so no need to add all TC synapses)
    pre_gids = np.concatenate([ca._bluepy_sim.target_gids, np.array(list(proj_spike_trains.keys()))])
    return pre_gids, proj_spike_trains, ca.config.bc.Run_Default.SpikeLocation


def run_sim(ssim, gid, pre_gids, pre_spike_trains, spike_loc):
    """Reruns simulation of a single `gid` w/ all the inputs from the network simulation"""
    # instantiate gid with replay on all synapses and the same input as it gets in the network simulation
    ssim.instantiate_gids([gid], add_synapses=True, add_projections=True, add_minis=True, intersect_pre_gids=pre_gids,
                          add_stimuli=True, add_replay=True, pre_spike_trains=pre_spike_trains)
    cell = ssim.cells[gid]
    all_sections = cell.cell.getCell().all

    # set up fake NetCon to detect spikes
    nc = cell.create_netcon_spikedetector(None, location=spike_loc)
    spike_vec = bglibpy.neuron.h.Vector()
    nc.record(spike_vec)
    # record voltage from all sections (not just the soma...) built in BGLibPy fn. ignore `record_dt`
    for section in all_sections:
        cell.add_recording("neuron.h." + section.name() + "(0.5)._ref_v", dt=ssim.record_dt)

    ssim.run()

    # return spike times as DF (more columns to be added later...)
    spike_times = np.array(spike_vec)
    spikes = pd.DataFrame(data=spike_times[spike_times > 0], columns=["spike_times"])
    # get voltage recordings from all sections and create DF
    data, columns = [], []
    for section in all_sections:
        columns.append(section.name().split(".")[1])
        data.append(cell.get_recording("neuron.h." + section.name() + "(0.5)._ref_v").reshape(-1, 1))
    vs = pd.DataFrame(data=np.hstack(data), columns=columns, index=cell.get_time())
    vs.index.name = "time"

    return spikes, vs


def run(config_path, seed, gid):
    """Sets up and reruns single cell BGLibPy with 3 conditions: baseline, passive dendrites, and NMDA blocked,
    and saves spikes and voltages of all sections from them"""

    config = Config(config_path)
    save_dir = os.path.join(config.root_path, "analyses", "rerun_results")
    utils.ensure_dir(save_dir)
    sim_path = utils.get_sim_path(config.root_path).loc[seed]
    L.info(" Instantiating %i from %s in BGLibPy " % (gid, sim_path))

    L.info(" Running sim w/ baseline conditions ")
    t1 = time.time()
    ssim = utils.get_bglibpy_ssim(sim_path)
    pre_gids, pre_spike_trains, spike_loc = get_gid_instantiation_vars(ssim)
    spikes, vs = run_sim(ssim, gid, pre_gids, pre_spike_trains, spike_loc)
    spikes["condition"] = "baseline"
    spikes.to_pickle(os.path.join(save_dir, "seed%s_a%i_spikes.pkl" % (seed, gid)))
    vs.to_pickle(os.path.join(save_dir, "seed%s_a%i_baseline_voltages.pkl" % (seed, gid)))
    ssim.delete()
    gc.collect()
    t2 = time.time()
    L.info(" Baseline sim. finished in: %s " % time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))

    '''TODO: add flags to `run_sim()` to run w/ different conditions
    L.info(" Running sim w/ passive dendrites ")
    ssim = utils.get_bglibpy_ssim(sim_path)
    spikes_, vs = run_sim(ssim, 8473, pre_gids, pre_spike_trains, spike_loc)
    spikes_["condition"] = "passive_dendrites"
    spikes = pd.concat([spikes, spikes_], ignore_index=True)
    spikes = spikes.sort_values("spike_times")
    spikes.to_pickle(os.path.join(save_dir, "seed%s_a%i_spikes.pkl" % (seed, gid)))
    vs.to_pickle(os.path.join(save_dir, "seed%s_a%i_passivedend_voltages.pkl" % (seed, gid)))
    ssim.delete()
    gc.collect()
    t3 = time.time()
    L.info(" Sim. w/ passive dendrites finished in: %s " % time.strftime("%H:%M:%S", time.gmtime(t3 - t2)))

    L.info(" Running sim w/ NMDA channels blocked ")
    ssim = utils.get_bglibpy_ssim(sim_path)
    spikes_, vs = run_sim(ssim, 8473, pre_gids, pre_spike_trains, spike_loc)
    spikes_["condition"] = "no_NMDA"
    spikes = pd.concat([spikes, spikes_], ignore_index=True)
    spikes = spikes.sort_values("spike_times")
    spikes.to_pickle(os.path.join(save_dir, "seed%s_a%i_spikes.pkl" % (seed, gid)))
    vs.to_pickle(os.path.join(save_dir, "seed%s_a%i_noNMDA_voltages.pkl" % (seed, gid)))
    ssim.delete()
    gc.collect()
    t4 = time.time()
    L.info(" Sim. w/ blocked NMDA channels finished in: %s " % time.strftime("%H:%M:%S", time.gmtime(t4 - t3)))
    '''


