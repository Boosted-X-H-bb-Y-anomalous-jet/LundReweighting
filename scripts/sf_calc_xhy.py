import sys, os
sys.path.insert(0, '')
sys.path.append("../")
from utils.Utils import *
import subprocess
import datetime
import numpy as np
from math import sqrt

def calc_mask(ak8jets,gen_parts_eta_phi):
    cutoff=0.8
    mask = []
    for i in range(len(gen_parts_eta_phi)):
        max_delta_r = 0
        jet_eta = ak8jets[i][1]
        jet_phi = ak8jets[i][2]
        for gen_part in gen_parts_eta_phi[i]:
            eta = gen_part[0]
            phi = gen_part[1]
            delta_phi = abs(jet_phi-phi)
            delta_eta = jet_eta-eta
            pi = 3.14159
            if(delta_phi>pi):
                delta_phi = delta_phi - pi
            delta_r = np.sqrt(delta_eta**2+delta_phi**2)
            if delta_r>max_delta_r:
                max_delta_r=delta_r
        if(max_delta_r>cutoff):
            mask.append(0)
        else:
            mask.append(1)
    return mask



def calc_SF(fname,f_ratio_name):
    f_sig = h5py.File(fname, "r")
    f_ratio = ROOT.TFile.Open(f_ratio_name)

    #Class to help read input dataset 
    d = Dataset(f_sig, dtype = 1)
    d.compute_obs()

    #The cut we will compute a SF for 'Y_vae_loss > 0.00005'
    tag_obs = 'Y_vae_loss'
    score_thresh = 0.00005


    LP_rw = LundReweighter(f_ratio = f_ratio)


    max_evts = 10000
    #max_evts = 30000 #Uncomment for real measurement
    if d.f['event_info'].shape[0]<max_evts:
        max_evts = d.f['event_info'].shape[0]
    score = getattr(d, tag_obs)[:max_evts]
    score_cut = score > score_thresh

    ################### Compute reweighting factors

    #PF candidates in the AK8 jet
    pf1_cands = d.f["jet1_PFCands"][:max_evts]
    pf2_cands = d.f["jet2_PFCands"][:max_evts]
    Y_idx = d.f["Y_idx"][:max_evts]
    pf_cands = [pf1_cands[i] if idx==0 else pf2_cands[i] for idx,i in zip(Y_idx,range(len(Y_idx)))] #Select pf candidates of the Y-candidate ak8 jet

    #Generator level quarks from hard process
    gen_parts = d.f['gen_info'][:max_evts]
    gen_parts_eta_phi = gen_parts[:,:,1:3]
    gen_parts_pdg_ids = gen_parts[:,:,3]

    ak8_jets_1 = d.f['jet_kinematics'][:max_evts][:,2:6]#2-6 jet1, 6-10 jet2
    ak8_jets_2 = d.f['jet_kinematics'][:max_evts][:,6:10]#2-6 jet1, 6-10 jet2
    ak8_jets = [ak8_jets_1[i] if idx==0 else ak8_jets_2[i] for idx,i in zip(Y_idx,range(len(Y_idx)))]  #Select ak8 jets that are Y candidates
    
    pf_cands = [arr.astype(np.float64) for arr in pf_cands]

    good_matching_mask = calc_mask(ak8_jets,gen_parts_eta_phi)
    ak8_jets = [d for d, m in zip(ak8_jets, good_matching_mask) if m == 1]
    pf_cands = [d for d, m in zip(pf_cands, good_matching_mask) if m == 1]
    gen_parts_eta_phi = [d for d, m in zip(gen_parts_eta_phi, good_matching_mask) if m == 1]
    gen_parts_pdg_ids = [d for d, m in zip(gen_parts_pdg_ids, good_matching_mask) if m == 1]
    score_cut = [d for d, m in zip(score_cut, good_matching_mask) if m == 1]

    nom_weights = d.f["sys_weights"][:max_evts,0]
    nom_weights = [d for d, m in zip(nom_weights, good_matching_mask) if m == 1]
    nom_weights = np.ones(len(pf_cands)) #If we want to have weights=1. for all events


    mask_eff = len(ak8_jets)/max_evts
    print(f"Good matching efficiency: {mask_eff}")

    # inspect_evt = 985
    # print(pf_cands[inspect_evt])
    # print(gen_parts_eta_phi[inspect_evt])
    # print(gen_parts_pdg_ids[inspect_evt])
    # print(ak8_jets[inspect_evt])
    # print(ak8_jets_1[inspect_evt])
    # print(ak8_jets_2[inspect_evt])
    
    LP_weights = LP_rw.get_all_weights(pf_cands, gen_parts_eta_phi, ak8_jets, gen_parts_pdg_ids = gen_parts_pdg_ids)

    #multiply Lund plane weights with nominal event weights
    for key in LP_weights.keys():
        if('nom' in key or 'up' in key or 'down' in key):
            if(isinstance(LP_weights[key], np.ndarray)) : LP_weights[key] *= nom_weights



    #Fraction of prongs that are not well matched to subjets (want this to be low)
    print("Bad match frac %.2f" % np.mean(LP_weights['bad_match']))
    #Fraction of prongs that are still not well matched after reclustering with varied number of prongs
    print("Reclustered bad match frac %.2f" % np.mean(LP_weights['reclust_still_bad_match']))


    ###### Use weights to compute efficiency of a cut

    #Efficiency of the cut in nominal MC
    eff_nom = np.average(score_cut, weights = nom_weights)

    #Efficiency of the cut after the Lund Plane reweighting
    eff_rw = np.average(score_cut, weights = LP_weights['nom'])

    #Nominal 'scale factor'
    SF = eff_rw / eff_nom

    print("Nominal efficiency %.3f, Corrected efficiency %.3f, SF (corrected / nom) %.3f" % (eff_nom, eff_rw, SF))

    #NOTE, because there is kinematic dependence to the correction, it is better to use corrected efficiency computed 
    #separately for each MC sample rather than a single 'SF'

    ######  Compute uncertainties on the efficiency from the various weight variations ##############

    #statistical and pt extrapolation uncertainties derived from 100 variations of the weights 
    #take std dev to determine unc

    nToys = LP_weights['stat_vars'].shape[1]
    eff_toys = []
    pt_eff_toys = []
    for i in range(nToys):
        eff = np.average(score_cut, weights = LP_weights['stat_vars'][:,i])
        eff_toys.append(eff)

        eff1 = np.average(score_cut, weights = LP_weights['pt_vars'][:,i])
        pt_eff_toys.append(eff1)

    #Compute stat and pt uncertainty based on variation in the toys
    toys_mean = np.mean(eff_toys)
    toys_std = np.std(eff_toys)
    pt_toys_mean = np.mean(pt_eff_toys)
    pt_toys_std = np.std(pt_eff_toys)

    #if mean of toys is biased, also include it as an unc (should be zero)
    eff_stat_unc = (abs(toys_mean - eff_rw)  + toys_std) 
    eff_pt_unc = (abs(pt_toys_mean - eff_rw) + pt_toys_std)

    print("Stat variation toys eff. avg %.3f, std dev %.3f" % (toys_mean, toys_std))
    print("Pt variation toys eff. avg %.3f, std dev %.3f" % (pt_toys_mean, pt_toys_std))

    #Other systematics come from up/down variations of the weights
    sys_keys = ['sys', 'bquark', 'prongs', 'unclust', 'distortion']
    sys_uncs = dict()

    for sys in sys_keys: sys_uncs[sys] = [0.,0.]

    #Compute difference in efficiency due to weight variations as uncertainty
    def get_uncs(cut, weights_up, weights_down, eff_baseline):
        eff_up =  np.average(cut, weights = weights_up)
        eff_down =  np.average(cut, weights = weights_down)

        unc_up = eff_up - eff_baseline
        unc_down = eff_down - eff_baseline 
        return unc_up, unc_down

    for sys in sys_keys:
        unc_up, unc_down = get_uncs(score_cut, LP_weights[sys + '_up'], LP_weights[sys + '_down'], eff_rw)
        sys_uncs[sys] = [unc_up, unc_down]


    #Print uncertainty breakdown
    eff_str = "Calibrated efficiency  is %.2f +/- %.2f (stat) +/- %.2f (pt)" % (eff_rw, eff_stat_unc, eff_pt_unc )
    tot_unc_up = tot_unc_down = eff_stat_unc**2 + eff_pt_unc**2

    for sys in sys_keys:
        eff_str += " %.2f/%.2f (%s)" % (sys_uncs[sys][0], sys_uncs[sys][1], sys)
        up_var = max(sys_uncs[sys][0], sys_uncs[sys][1])
        down_var = min(sys_uncs[sys][0], sys_uncs[sys][1])
        tot_unc_up += up_var**2
        tot_unc_down += down_var**2



    tot_unc_up = tot_unc_up**0.5
    tot_unc_down = tot_unc_down**0.5

    #Print final calibrated efficiency and total uncertaintiy
    eff_str += "\n Original %.2f, Calibrated %.2f +%.2f/-%.2f \n"  % (eff_nom, eff_rw, tot_unc_up, tot_unc_down)

    print(eff_str)
    f_ratio.Close()
    return eff_nom,eff_rw,tot_unc_up,tot_unc_down



def xrdcp_merged_file(year,process):
    print(f"Calculating SF for {process} {year}")
    h5_dir  = f"/store/user/roguljic/H5_output/{year}/{process}/"
    xrdcp_cmd = f"xrdcp root://cmseos.fnal.gov/{h5_dir}/merged.h5 ."
    subprocess.call(xrdcp_cmd,shell=True)


year = "2018"
process = "MX1600_MY90"
xrdcp_merged_file(year,process)
eff_nom,eff_rw,tot_unc_up,tot_unc_down = calc_SF("merged.h5","data/ratio_2018.root")
sf_string = "..."
current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
line_to_write = f"{sf_string} - {current_date}\n"

with open("SFs.txt", "a") as file:
    file.write(line_to_write)
subprocess.call("rm merged.h5",shell=True)

