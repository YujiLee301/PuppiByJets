"""
script to test the physics performance of a given model
"""
import math
import scipy.stats
from collections import OrderedDict
from pyjet import cluster, DTYPE_PTEPM
import argparse
import torch
from torch_geometric.data import DataLoader
import models as models
import utils
import matplotlib
from copy import deepcopy
import os
import copy
import uproot
import awkward as ak

# matplotlib.use("pdf")
import numpy as np
import random
import pickle
import joblib
from timeit import default_timer as timer
from tqdm import tqdm

import matplotlib as mpl
import imageio

# mpl.use("pdf")
import matplotlib.pyplot as plt
import mplhep as hep

hep.set_style(hep.style.CMS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
testneu = 1
# Options to Chg+Neu or Chg only
def SetPxPyPzE(pt, eta, phi):
    
    Px = pt * math.cos(phi)
    Py = pt * math.sin(phi)
    Pz = pt * math.sinh(eta)
    Energy = math.sqrt(Px*Px + Py*Py + Pz*Pz)
    return Px, Py, Pz, Energy

def GetJetInvMass(pt, eta, phi):
    Px = 0
    Py = 0
    Pz = 0
    Energy = 0
    for i in range(len(pt)):
        px, py, px, ene = SetPxPyPzE(pt[i], eta[i], phi[i])
        Px+=px
        Py+=py
        Pz+=Pz
        Energy+=ene
    
    return math.sqrt(Energy*Energy-Px*Px-Py*Py-Pz*Pz)

def generate_jet_mass_truth(dataset):
    # calculate Genjet inv mass for each graph
    for graph in dataset:
        pt_truth = np.array(graph.GenPart_nump[:, 2].cpu().detach())
        eta_truth = np.array(graph.GenPart_nump[:, 0].cpu().detach())
        phi_truth = np.array(graph.GenPart_nump[:, 1].cpu().detach())
        graph.JetmassTruth = GetJetInvMass(pt_truth, eta_truth, phi_truth)
    
    return dataset


class Args(object):
    """
    arguments for loading models
    """

    def __init__(self, model_type='Gated', do_boost=False, extralayers=False):
        self.model_type = model_type
        self.num_layers = 3
        self.batch_size = 1
        self.hidden_dim = 20
        self.dropout = 0
        self.opt = 'adam'
        self.weight_decay = 0
        self.lr = 0.01
        self.do_boost = do_boost
        self.extralayers = extralayers



def postProcessing(data, preds):
    """
    reconstruct jet and MET,
    compare the reco-ed jet and MET with truth ones,
    using the input data and ML weights (pred)
    """
    pt = np.array(data.x[:, 2].cpu().detach())
    eta = np.array(data.x[:, 0].cpu().detach())
    phi = np.array(data.x[:, 1].cpu().detach())
    puppi = np.array(data.pWeight.cpu().detach())
    puppichg = np.array(data.pWeightchg.cpu().detach())
    # puppi = np.array(data.x[:,data.num_feature_actual[0].item()-1].cpu().detach())
    # truth = np.array(data.y.cpu().detach())
    
    JetmassTruth = data.JetmassTruth
    
    # remove pt < 0.5 particles
    pt[pt < 0.01] = 0

    # apply CHS to puppi weights
    charge_index = data.Charge_index[0]
    neutral_index = data.Neutral_index[0]
    
    lv_index = data.LV_index[0]
    pu_index = data.PU_index[0]

    chglv_index = list(set(lv_index) & set(charge_index))
    chgpu_index = list(set(pu_index) & set(charge_index))
    
    if testneu == 1:
        chargeOnly = 0
    else:
        chargeOnly = 1

    if chargeOnly == 1 :
       pt[neutral_index] = 0

    # puppi information
    if testneu == 1:
        puppichg[charge_index] = puppi[charge_index]
    if testneu == 0:
        pt_puppi = pt * puppichg
    if testneu == 1:
        pt_puppi = pt * puppi
    pt_CHS = pt * puppi
    if testneu == 1:
        pt_CHS[neutral_index] = pt[neutral_index]
    # apply some weight cuts on puppi
    cut = 0.41  # GeV
    wcut = 0.17
    cut = 0.0
    wcut = 0.0
    pt_puppi_wcut = np.array(pt, copy=True)
    pt_puppi_wcut[(puppi < wcut) | (pt_puppi < cut)] = 0.


    Jet_mass_puppi = GetJetInvMass(pt_puppi, eta, phi)
    mass_diff_puppi = (Jet_mass_puppi - JetmassTruth)/JetmassTruth

    Jet_mass_puppi_wcut = GetJetInvMass(pt_puppi_wcut, eta, phi)
    mass_diff_puppi_wcut = (Jet_mass_puppi_wcut - JetmassTruth)/JetmassTruth

    Jet_mass_CHS = GetJetInvMass(pt_CHS, eta, phi)
    mass_diff_CHS = (Jet_mass_CHS - JetmassTruth)/JetmassTruth

    neu_pred = []
    neu_puppi = []
    chlv_pred = []
    chpu_pred = []
    chlv_puppi = []
    chpu_puppi = []

    for pred in preds:
        # print("preds: ", pred)
        pred = np.array(pred[0][:, 0].cpu().detach())
        #pred[pred<0.3] = 0
        predcopy = pred
        predcopyA = []
        for j in range(len(predcopy)):
            predcopyA.append(predcopy[j])

        predcopy[charge_index] = -2
        for m in range(len(predcopy)):
            if predcopy[m]>-0.1:
                neu_pred.append(predcopy[m])
                neu_puppi.append(puppichg[m])
        
        predcopyA = np.array(predcopyA)
        #predcopyA[predcopyA<0.3] = 0
        #predcopyA[predcopyA>0.3] = 1


        for mi in chglv_index:
            chlv_pred.append(predcopyA[mi])
            chlv_puppi.append(puppichg[mi])
        for mj in chgpu_index:
            chpu_pred.append(predcopyA[mj])
            chpu_puppi.append(puppichg[mj])
        # apply CHS to predictions
        # charge_index = data.Charge_index[0]
        #pred[charge_index] = puppichg[charge_index]
        if testneu == 1:
            predcopyA[charge_index] = puppi[charge_index]
        
        pt_pred = pt * predcopyA
        Jet_mass_pred = GetJetInvMass(pt_pred, eta, phi)

        mass_diff_pred = (Jet_mass_pred - JetmassTruth)/JetmassTruth
      
    
    

    return mass_diff_CHS, mass_diff_puppi, mass_diff_puppi_wcut, mass_diff_pred, neu_pred, neu_puppi, chlv_pred, chpu_pred, chlv_puppi, chpu_puppi


def test(filelists, models={}):

    for model in models.values():
        model.to('cuda:0')
        model.eval()

    neu_weight = []
    neu_puppiweight = []
    chlv_weight = []
    chpu_weight = []
    chlv_puppiweight = []
    chpu_puppiweight = []

    ievt = 0
    for ifile in filelists:
        print("ifile: ", ifile)
        fp = open(ifile, "rb")
        dataset = joblib.load(fp)
        generate_jet_mass_truth(dataset)
        data = DataLoader(dataset, batch_size=1)
        loader = data

        

        for data in loader:
            ievt += 1
            # if ievt > 10:
            #    break

            if ievt % 100 == 0:
                print("processed {} events".format(ievt))
            with torch.no_grad():
                data = data.to(device)
                # max(dim=1) returns values, indices tuple; only need indices

                # loop over model in models and run the inference
                preds = []

                for model in models.values():
                    model.to('cuda:0')
                    model.eval()

                    pred = model.forward(data)
                    # print("pred here: ", pred)
                    preds.append(pred)

                mass_diff_CHS, mass_diff_puppi, mass_diff_puppi_wcut, mass_diff_pred, neus_pred, neus_puppi, chlvs_pred, chpus_pred, chlvs_puppi, chpus_puppi = postProcessing(
                    data, preds)
               
                for m0 in range(len(neus_pred)):
                    neu_puppiweight.append(neus_puppi[m0])
                for m1 in range(len(neus_pred)):
                    neu_weight.append(neus_pred[m1])
                for m2 in range(len(chlvs_pred)):
                    chlv_weight.append(chlvs_pred[m2])
                for m3 in range(len(chpus_pred)):
                    chpu_weight.append(chpus_pred[m3])
                for m4 in range(len(chlvs_puppi)):
                    chlv_puppiweight.append(chlvs_puppi[m4])
                for m5 in range(len(chpus_puppi)):
                    chpu_puppiweight.append(chpus_puppi[m5])
                
        print("eventNum:"+str(ievt))

        fp.close()

    return  mass_diff_CHS, mass_diff_puppi, mass_diff_puppi_wcut, mass_diff_pred,  neu_weight, neu_puppiweight, chlv_weight, chpu_weight, chlv_puppiweight, chpu_puppiweight


def main(modelname, filelists):
    # load models
    args = Args()
    model_gated_boost = models.GNNStack(9, args.hidden_dim, 1, args)
    # model_load.load_state_dict(torch.load('best_valid_model_semi.pt'))
    model_gated_boost.load_state_dict(torch.load(modelname))

    modelcolls = OrderedDict()
    modelcolls['gated_boost'] = model_gated_boost

    # run the tests
    #filelists = ["../data_pickle/dataset_graph_puppi_test_40004000"]
    mass_diff_CHS, mass_diff_puppi, mass_diff_puppi_wcut, mass_diff_pred, neu_weight, neu_puppiweight, chlv_weight, chpu_weight, chlv_puppiweight, chpu_puppiweight = test(
        filelists, modelcolls)

    # plot the differences
    def getResol(input):
        return (np.quantile(input, 0.84) - np.quantile(input, 0.16))/2

    def getStat(input):
        return float(np.median(input)), float(getResol(input))



    linewidth = 1.5
    fontsize = 18

   #  %matplotlib inline
    plt.style.use(hep.style.ROOT)
    fig = plt.figure(figsize=(10, 8))
    mass_diff = mass_diff_pred
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='blue', linewidth=linewidth,
             density=True, label=r'Semi-supervised, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    mass_diff = mass_diff_puppi
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='green', linewidth=linewidth, 
             density=True, label=r'PUPPI, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    mass_diff = mass_diff_puppi_wcut
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='red', linewidth=linewidth, 
             density=True, label=r'PF, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    mass_diff = mass_diff_CHS
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='orange', linewidth=linewidth, 
             density=True, label=r'CHS, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    # plt.xlim(-1.0,1.3)
    plt.xlabel(r"Jet Mass $(m_{reco} - m_{truth})/m_{truth}$")
    plt.ylabel('density')
    plt.ylim(0, 6)
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.savefig("Jet_mass_diff.pdf")
    plt.show()


    fig = plt.figure(figsize=(10, 8))
    neutral_weight_total = np.array(neu_weight)
    neutral_puweight_total = np.array(neu_puppiweight)
    chlv_weight_total = np.array(chlv_weight)
    chpu_weight_total = np.array(chpu_weight)
    chlv_puweight_total = np.array(chlv_puppiweight)
    chpu_puweight_total = np.array(chpu_puppiweight)
    print(chlv_weight_total[:100])
    neutral_weight_total = np.array(neu_weight)
    plt.hist(neutral_weight_total, bins=40, range=(0, 1), histtype='step', color='blue', linewidth=linewidth,
              density=True, label=r'Neutral particle weight')
    plt.hist(chlv_weight_total, bins=40, range=(0, 1), histtype='step', color='green', linewidth=linewidth,
              density=True, label=r'Charged LV particle weight')
    plt.hist(chpu_weight_total, bins=40, range=(0, 1), histtype='step', color='pink', linewidth=linewidth,
              density=True, label=r'Charged PU particle weight')
    
    
    plt.xlabel(r"SSL weight")
    plt.ylabel('density')
    plt.legend()
    plt.show()
    plt.savefig("GNNweight.pdf")

    fig = plt.figure(figsize=(10, 8))
    plt.hist(neutral_puweight_total, bins=40, range=(0, 1), histtype='step', color='blue', linewidth=linewidth,
              density=True, label=r'Neutral particle weight')
    plt.hist(chlv_puweight_total, bins=40, range=(0, 1), histtype='step', color='green', linewidth=linewidth,
              density=True, label=r'Charged LV particle weight')
    plt.hist(chpu_puweight_total, bins=40, range=(0, 1), histtype='step', color='pink', linewidth=linewidth,
              density=True, label=r'Charged PU particle weight')
    plt.xlabel(r"puppi weight")
    plt.ylabel('density')
    plt.legend()
    plt.show()
    plt.savefig("GNNpuppi.pdf")



    # more plots to be included


if __name__ == '__main__':
    modelname = "test/best_valid_model_nPU20_deeper.pt"
    filelists = ["../data_pickle/dataset_graph_puppi_test_Wjets4000"]
    main(modelname, filelists)