import sys, os
import subprocess
import numpy as np
import h5py
import ROOT
import mplhep as hep
import matplotlib.pyplot as plt

sys.path.insert(0, '')
sys.path.append("../")
from utils.Utils import *

def calculate_weights_and_observables(fname, f_ratio_name,max_evts=10000):
    f_sig = h5py.File(fname, "r")
    f_ratio = ROOT.TFile.Open(f_ratio_name)

    d = Dataset(f_sig, dtype=1)
    d.compute_obs()

    LP_rw = LundReweighter(f_ratio=f_ratio)

    if d.f['event_info'].shape[0] < max_evts:
        max_evts = d.f['event_info'].shape[0]

    Y_idx = d.f["Y_idx"][:max_evts]
    ak8_jets_1 = d.f['jet_kinematics'][:max_evts][:, 2:6]
    ak8_jets_2 = d.f['jet_kinematics'][:max_evts][:, 6:10]
    ak8_jets = [ak8_jets_1[i] if idx == 0 else ak8_jets_2[i] for idx, i in zip(Y_idx, range(len(Y_idx)))]

    jet_masses = [ak8_jets_1[i][3] if idx == 0 else ak8_jets_2[i][3] for idx, i in zip(Y_idx, range(len(Y_idx)))]
    dijet_masses = d.f['jet_kinematics'][:max_evts][:, 0]

    pf1_cands = d.f["jet1_PFCands"][:max_evts]
    pf2_cands = d.f["jet2_PFCands"][:max_evts]
    pf_cands = [pf1_cands[i] if idx == 0 else pf2_cands[i] for idx, i in zip(Y_idx, range(len(Y_idx)))]
    pf_cands = [arr.astype(np.float64) for arr in pf_cands]

    gen_parts = d.f['gen_info'][:max_evts]
    gen_parts_eta_phi = gen_parts[:, :, 1:3]
    gen_parts_pdg_ids = gen_parts[:, :, 3]

    LP_weights = LP_rw.get_all_weights(pf_cands, gen_parts_eta_phi, ak8_jets, gen_parts_pdg_ids=gen_parts_pdg_ids, normalize=False)
    lp_weights_nom = LP_weights['nom']

    f_ratio.Close()

    return jet_masses, dijet_masses, lp_weights_nom

def plot_jet_masses(jet_masses, weights, output_plot):
    plt.style.use(hep.style.CMS)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    mass_range = (60, 400)
    bins = 34

    counts_before, bin_edges, _ = ax1.hist(jet_masses, bins=bins, range=mass_range, label="Before Weighting", color='blue', density=True, histtype='step', facecolor='none')
    counts_after, _, _ = ax1.hist(jet_masses, bins=bins, range=mass_range, weights=weights, label="After Weighting", color='red', density=True, histtype='step', facecolor='none')

    max_value = max(max(counts_before), max(counts_after))
    ax1.set_ylim(0, 1.3 * max_value)
    hep.cms.label("WiP", data=False, year=year, ax=ax1)

    ax1.set_ylabel("Normalized Events")
    ax1.legend(loc='upper right')

    ratio = np.divide(counts_after, counts_before, out=np.zeros_like(counts_after), where=counts_before != 0)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])


    ax2.plot(bin_centers, ratio, marker='o', linestyle='None', color='black')
    ax2.set_ylabel('Ratio')
    ax2.set_xlabel('Jet Mass [GeV]')
    ax2.axhline(1, color='gray', lw=1, linestyle='--')
    ax2.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    print(f"Jet mass plot saved as {output_plot}")

def plot_dijet_masses(dijet_masses, weights, output_plot):
    plt.style.use(hep.style.CMS)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    mass_range = (1200, 3000)
    bins = 36

    counts_before, bin_edges, _ = ax1.hist(dijet_masses, bins=bins, range=mass_range, label="Before Weighting", color='blue', density=True, histtype='step', facecolor='none')
    counts_after, _, _ = ax1.hist(dijet_masses, bins=bins, range=mass_range, weights=weights, label="After Weighting", color='red', density=True, histtype='step', facecolor='none')
    max_value = max(max(counts_before), max(counts_after))
    ax1.set_ylim(0, 1.3 * max_value)

    hep.cms.label("WiP", data=False, year=year, ax=ax1)

    ax1.set_ylabel("Normalized Events")
    ax1.legend(loc='upper right')

    ratio = np.divide(counts_after, counts_before, out=np.zeros_like(counts_after), where=counts_before != 0)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    ax2.plot(bin_centers, ratio, marker='o', linestyle='None', color='black')
    ax2.set_ylabel('Ratio')
    ax2.set_xlabel('Dijet Mass [GeV]')
    ax2.axhline(1, color='gray', lw=1, linestyle='--')
    ax2.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    print(f"Dijet mass plot saved as {output_plot}")

def xrdcp_merged_file(year, process):
    print(f"Processing {process} for {year}")
    h5_dir = f"/store/user/roguljic/H5_output/{year}/{process}/"
    xrdcp_cmd = f"xrdcp -f root://cmseos.fnal.gov/{h5_dir}/merged.h5 ."
    subprocess.call(xrdcp_cmd, shell=True)

max_evts = 30000
year = "2018"
processes = ["TTToHadronic","MX2200_MY300","MX2200_MY125"]
files = {"TTToHadronic":"merged_ttbar_2018.h5","MX2200_MY300":"mx2200_my300.h5","MX2200_MY125":"mx2200_my125.h5"}
for process in processes:
    h5_file = files[process]
    ratio_file = f"data/ratio_{year}.root"
    jet_masses, dijet_masses, lp_weights_nom = calculate_weights_and_observables(h5_file, ratio_file,max_evts=max_evts)

    output_plot_jet_mass = f"jet_masses_{process}_{year}.png"
    plot_jet_masses(jet_masses, lp_weights_nom, output_plot_jet_mass)

    output_plot_dijet_mass = f"dijet_masses_{process}_{year}.png"
    plot_dijet_masses(dijet_masses, lp_weights_nom, output_plot_dijet_mass)
