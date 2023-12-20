import math
from tqdm import tqdm
from collections import OrderedDict
from timeit import default_timer as timer
import pickle
import random
import numpy as np
import argparse
import torch
from torch_geometric.data import DataLoader
import models as models
import utils
import test_physic_metrics as phym
import matplotlib
from copy import deepcopy
import os
import sys
import matplotlib.pyplot as plt
import mplhep as hep

hep.set_style(hep.style.CMS)

matplotlib.use("pdf")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--pulevel', type=int,
                        help='pileup level for the dataset')
    parser.add_argument('--training_path', type=str,
                        help='path for training graphs')
    parser.add_argument('--validation_path', type=str,
                        help='path for validation graphs')
    parser.add_argument('--save_dir', type=str,
                        help='directory to save trained model and plots')

    parser.set_defaults(model_type='Gated',
                        num_layers=6,
                        batch_size=1,
                        hidden_dim=30,
                        dropout=0.3,
                        opt='adam',
                        weight_decay=0,
                        lr=0.0001,
                        pulevel=80,
                        training_path="../data_pickle/dataset_graph_puppi_WjetsDR83000",
                        validation_path="../data_pickle/dataset_graph_puppi_val_WjetsDR81000",
                        save_dir="test",
                        )

    return parser.parse_args()

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
        px, py, pz, ene = SetPxPyPzE(pt[i], eta[i], phi[i])
        Px+=px
        Py+=py
        Pz+=pz
        Energy+=ene
    
    return math.sqrt(Energy*Energy-Px*Px-Py*Py-Pz*Pz)

def GetJetInvMass_Tensor(pt, eta, phi):
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px*px + py * py + pz*pz)

    Px = torch.sum(px)
    Py = torch.sum(py)
    Pz = torch.sum(pz)
    Energy = torch.sum(energy)
    mass = torch.sqrt(Energy*Energy-Px*Px-Py*Py-Pz*Pz)

    return torch.reshape(mass,[1])


def train(dataset, dataset_validation, args, batchsize):
    directory = args.save_dir
    # parent_dir = "/home/gpaspala/new_Pileup_GNN/Pileup_GNN/fast_simulation/"
    # path = os.path.join(parent_dir, directory)
    path = directory
    isdir = os.path.isdir(path)

    if isdir == False:
        os.mkdir(path)

    start = timer()

    

    training_loader = DataLoader(dataset, batch_size=batchsize)
    validation_loader = DataLoader(dataset_validation, batch_size=batchsize)
    model = models.GNNStack(
        dataset[0].num_features, args.hidden_dim, 1, args)
    model.to(device)
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    

    # train
    #
    # todo: this bunch of lists can be replaced with a small class or so
    #
    epochs_train = []
    epochs_valid = []
    loss_graph = []
    loss_graph_train = []
    loss_graph_valid = []
    loss_graph_puppi = []
    loss_graph_pf = []

    train_graph_SSLMassdiffMu = []
    train_graph_PUPPIMassdiffMu = []
    train_graph_SSLMassSigma = []
    train_graph_PUPPIMassSigma = []
    valid_graph_SSLMassdiffMu = []
    valid_graph_PUPPIMassdiffMu = []
    valid_graph_SSLMassSigma = []
    valid_graph_PUPPIMassSigma = []

    best_validation_SSLMassSigma = 0.99
    best_valid_SSLMassdiffMu = 0.99

    count_event = 0
    converge = False
    converge_num_event = 0
    last_steady_event = 0
    lowest_valid_loss = 10

    while converge == False:
        model.train()

        #t = tqdm(total=len(training_loader), colour='green',
        #         bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        loss_avg = utils.RunningAverage()
        for batch in training_loader:
            count_event += 1
            epochs_train.append(count_event)
            batch = batch.to(device)
            pred, xa_ = model.forward(batch)
            pred_ = xa_.cpu().detach()
            pt = batch.x[:, 2].cpu().detach()
            eta = batch.x[:, 0].cpu().detach()
            phi = batch.x[:, 1].cpu().detach()
            pt_ =np.array(pt)
            eta_ = np.array(eta)
            phi_ =np.array(phi)
            loss = model.loss(batch,pred)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_graph.append(loss)
            # print("cur_loss ", cur_loss)
            loss_avg.update(loss)
            # print("loss_avg ", loss_avg())
            #t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            #t.update()

            if count_event % 200 == 0:

                modelcolls = OrderedDict()
                modelcolls['gated_boost'] = model
                training_loss,training_loss_puppi,training_loss_pf,train_SSLMassdiffMu, \
                    train_SSLMassSigma, train_PUPPIMassdiffMu, train_PUPPIMassSigma = test(
                        training_loader, model, 0, count_event, args, modelcolls, args.training_path)

                valid_loss,valid_loss_puppi,valid_loss_pf, valid_SSLMassdiffMu, \
                    valid_SSLMassSigma, valid_PUPPIMassdiffMu, valid_PUPPIMassSigma = test(
                        validation_loader, model, 1, count_event, args, modelcolls, args.validation_path)

                epochs_valid.append(count_event)
                loss_graph_valid.append(valid_loss)
                loss_graph_train.append(training_loss)
                loss_graph_puppi.append(valid_loss_puppi)
                loss_graph_pf.append(valid_loss_pf)

                train_graph_SSLMassdiffMu.append(train_SSLMassdiffMu)
                train_graph_PUPPIMassdiffMu.append(train_PUPPIMassdiffMu)
                train_graph_SSLMassSigma.append(train_SSLMassSigma)
                train_graph_PUPPIMassSigma.append(train_PUPPIMassSigma)

                valid_graph_SSLMassdiffMu.append(valid_SSLMassdiffMu)
                valid_graph_PUPPIMassdiffMu.append(valid_PUPPIMassdiffMu)
                valid_graph_SSLMassSigma.append(valid_SSLMassSigma)
                valid_graph_PUPPIMassSigma.append(valid_PUPPIMassSigma)
                
                
                if (valid_SSLMassSigma/(1.01-abs(valid_SSLMassdiffMu))) < (best_validation_SSLMassSigma/(1.01-abs(best_valid_SSLMassdiffMu))):
                    best_validation_SSLMassSigma = valid_SSLMassSigma 
                    best_valid_SSLMassdiffMu = valid_SSLMassdiffMu 
                    print("model is saved in " + path + "/best_valid_model_Zjets.pt")
                    if isinstance(model, torch.nn.DataParallel):
                         model_state_dict = model.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()
                        torch.save(model_state_dict, path +
                                "/best_valid_model_Zjets.pt")

                if valid_loss >= lowest_valid_loss:
                    print(
                        "valid loss increase at event " + str(count_event) + " with validation loss " + str(valid_loss))
                    if last_steady_event == count_event - 1000:
                        converge_num_event += 1
                        if converge_num_event > 30:
                            converge = True
                            break
                        else:
                            last_steady_event = count_event
                    else:
                        converge_num_event = 1
                        last_steady_event = count_event
                    # print("converge num event " + str(converge_num_event))
                else:
                    print("lowest valid loss " + str(valid_loss))
                    lowest_valid_loss = valid_loss

                if count_event == 2000:
                    converge = True
                    break

        #t.close()

    end = timer()
    training_time = end - start
    print("training time " + str(training_time))
    
    loss_graph_valid_ = []
    loss_graph_train_ = []
    loss_graph_puppi_ = []
    loss_graph_pf_ = []
    train_graph_SSLMassdiffMu_ = []
    valid_graph_SSLMassdiffMu_ = []
    train_graph_PUPPIMassdiffMu_ = []
    train_graph_SSLMassSigma_ = []
    valid_graph_SSLMassSigma_ = []
    train_graph_PUPPIMassSigma_ = []
    for loss_graph_valid_i in loss_graph_valid:
        loss_graph_valid_value = loss_graph_valid_i.cpu().detach()
        loss_graph_valid_.append(loss_graph_valid_value.item())
    for loss_graph_train_i in loss_graph_train:
        loss_graph_train_value = loss_graph_train_i.cpu().detach()
        loss_graph_train_.append(loss_graph_train_value.item())
    for loss_graph_puppi_i in loss_graph_puppi:
        loss_graph_puppi_value = loss_graph_puppi_i.cpu().detach()
        loss_graph_puppi_.append(loss_graph_puppi_value.item())
    for loss_graph_pf_i in loss_graph_pf:
        loss_graph_pf_value = loss_graph_pf_i.cpu().detach()
        loss_graph_pf_.append(loss_graph_pf_value.item())

    for train_graph_SSLMassdiffMu_i in train_graph_SSLMassdiffMu:
        train_graph_SSLMassdiffMu_value = train_graph_SSLMassdiffMu_i
        train_graph_SSLMassdiffMu_.append(train_graph_SSLMassdiffMu_value)
    for valid_graph_SSLMassdiffMu_i in valid_graph_SSLMassdiffMu:
        valid_graph_SSLMassdiffMu_value = valid_graph_SSLMassdiffMu_i
        valid_graph_SSLMassdiffMu_.append(valid_graph_SSLMassdiffMu_value)
    for train_graph_PUPPIMassdiffMu_i in train_graph_PUPPIMassdiffMu:
        train_graph_PUPPIMassdiffMu_value = train_graph_PUPPIMassdiffMu_i
        train_graph_PUPPIMassdiffMu_.append(train_graph_PUPPIMassdiffMu_value)
    for train_graph_SSLMassSigma_i in train_graph_SSLMassSigma:
        train_graph_SSLMassSigma_value = train_graph_SSLMassSigma_i
        train_graph_SSLMassSigma_.append(train_graph_SSLMassSigma_value)
    for valid_graph_SSLMassSigma_i in valid_graph_SSLMassSigma:
        valid_graph_SSLMassSigma_value = valid_graph_SSLMassSigma_i
        valid_graph_SSLMassSigma_.append(valid_graph_SSLMassSigma_value)
    for train_graph_PUPPIMassSigma_i in train_graph_PUPPIMassSigma:
        train_graph_PUPPIMassSigma_value = train_graph_PUPPIMassSigma_i
        train_graph_PUPPIMassSigma_.append(train_graph_PUPPIMassSigma_value)
    #To-do plotting loss here
    plt.figure()
    plt.plot(epochs_valid, loss_graph_valid_, label = 'valid', linestyle = 'solid', linewidth = 1, color = 'b')
    plt.plot(epochs_valid, loss_graph_train_, label = 'train', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_valid, loss_graph_puppi_, label = 'valid PUPPI', linestyle = 'solid', linewidth = 1, color = 'r')
    plt.plot(epochs_valid, loss_graph_pf_, label = 'valid PF', linestyle = 'solid', linewidth = 1, color = 'orange')
    #plt.plot(epochs_valid, valid_graph_PUPPIMassdiffMu, label = 'PUPPI_valid_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.savefig(args.save_dir + "/loss.pdf")
    plt.close()

    plt.figure()
    plt.plot(epochs_valid, train_graph_SSLMassdiffMu_, label = 'Semi-supervised_train_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_valid, valid_graph_SSLMassdiffMu_, label = 'Semi-supervised_valid_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'b')
    plt.plot(epochs_valid, train_graph_PUPPIMassdiffMu_, label = 'PUPPI_train_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'r')
    #plt.plot(epochs_valid, valid_graph_PUPPIMassdiffMu, label = 'PUPPI_valid_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('mean diff')
    plt.legend(loc=4)
    plt.savefig(args.save_dir + "/Jet_mass_diff_mean.pdf")
    plt.close()

    plt.figure()
    plt.plot(epochs_valid, train_graph_SSLMassSigma_, label = 'Semi-supervised_train_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_valid, valid_graph_SSLMassSigma_, label = 'Semi-supervised_valid_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'b')
    plt.plot(epochs_valid, train_graph_PUPPIMassSigma_, label = 'PUPPI_train_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'r')
    #plt.plot(epochs_valid, valid_graph_PUPPIMassSigma, label = 'PUPPI_valid_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('sigma diff')
    plt.legend(loc=4)
    plt.savefig(args.save_dir + "/Jet_mass_diff_sigma.pdf")
    plt.close()


  


def test(loader, model, indicator, epoch, args, modelcolls, pathname):
    if indicator == 0:
        postfix = 'Train'
    elif indicator == 1:
        postfix = 'Validation'
    else:
        postfix = 'Test'

    model.eval()

    pred_all = None
    pred_hybrid_all = None
    label_all = None
    puppi_all = None
    test_mask_all = None
    mask_all_neu = None
    total_loss = 0
    total_loss_puppi = 0
    total_loss_pf = 0
    count = 0
    neu_pred = []
    chlv_pred = []
    chpu_pred = []
    for data in loader:
        count += 1
        if count == epoch and indicator == 0:
            break
        with torch.no_grad():
            num_feature = data.num_features
            
            data = data.to(device)
            # max(dim=1) returns values, indices tuple; only need indices
            pred, pred_hybrid = model.forward(data)
            eta = data.x[:, 0]
            phi = data.x[:, 1]
            pt = data.x[:, 2]
            pred_hybrid_ = np.array(pred_hybrid.cpu().detach())
            charge_index = data.Charge_index[0]
            neutral_index = data.Neutral_index[0]
    
            lv_index = data.LV_index[0]
            pu_index = data.PU_index[0]

            chglv_index = list(set(lv_index) & set(charge_index))
            chgpu_index = list(set(pu_index) & set(charge_index))
            predcopy = pred_hybrid_
            predcopyA = []
            for j in range(len(predcopy)):
               predcopyA.append(predcopy[j])
            predcopyA = np.array(predcopyA)
            for mi in chglv_index:
                chlv_pred.append(predcopyA[mi])
            for mj in chgpu_index:
                chpu_pred.append(predcopyA[mj])
            predcopy[charge_index] = -2
            for m in range(len(predcopy)):
                if predcopy[m]>-0.1:
                    neu_pred.append(predcopy[m])
            
            
            # puppi = data.x[:, data.num_feature_actual[0].item() - 1]
            puppi = data.pWeight
            total_loss += model.loss(data, pred)
            total_loss_puppi += model.loss(data, GetJetInvMass_Tensor(pt*puppi,eta,phi))
            total_loss_pf += model.loss(data, GetJetInvMass_Tensor(pt,eta,phi))
               

    if indicator == 0:
        total_loss /= min(epoch, len(loader.dataset))
        total_loss_puppi /= min(epoch, len(loader.dataset))
        total_loss_pf /= min(epoch, len(loader.dataset))
    else:
        total_loss /= len(loader.dataset)
        total_loss_puppi /= len(loader.dataset)
        total_loss_pf /= len(loader.dataset)
    
    filelists = []
    filelists.append(pathname)

    mass_diff_CHS, mass_diff_puppi, mass_diff_puppi_wcut, mass_diff_pred, neu_weight, neu_puppiweight, chlv_weight, chpu_weight, chlv_puppiweight, chpu_puppiweight  =phym.test(
        filelists, modelcolls)

    # plot the differences
    def getResol(input):
        return (np.quantile(input, 0.84) - np.quantile(input, 0.16))/2

    def getStat(input):
        return float(np.median(input)), float(getResol(input))

    sub_dir = "prob_plots"
    parent_dir = "./" + args.save_dir

    path = os.path.join(parent_dir, sub_dir)

    isdir = os.path.isdir(path)
    if isdir == False:
        os.mkdir(os.path.join(parent_dir, sub_dir))

    linewidth = 1.5
    fontsize = 18
   #  %matplotlib inline
    plt.style.use(hep.style.ROOT)
    fig = plt.figure(figsize=(10, 8))
    mass_diff_pred_ = []
    mass_diff_puppi_ = []
    mass_diff_puppi_wcut_ = []
    mass_diff_CHS_ = []
    for mass_diff_i in mass_diff_pred:
        mass_diff_pred_value = mass_diff_i
        mass_diff_pred_.append(mass_diff_pred_value)
    for mass_diff_i in mass_diff_puppi:
        mass_diff_puppi_value = mass_diff_i.cpu().detach()
        mass_diff_puppi_.append(mass_diff_puppi_value[0])
    for mass_diff_i in mass_diff_puppi_wcut:
        mass_diff_puppi_wcut_value = mass_diff_i.cpu().detach()
        mass_diff_puppi_wcut_.append(mass_diff_puppi_wcut_value[0])
    for mass_diff_i in mass_diff_CHS:
        mass_diff_CHS_value = mass_diff_i.cpu().detach()
        mass_diff_CHS_.append(mass_diff_CHS_value[0])
    
    mass_diff = np.array(mass_diff_pred_)
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='blue', linewidth=linewidth,
             density=True, label=r'Semi-supervised, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    SSLMassdiffMu, SSLMassSigma = getStat(mass_diff)
    mass_diff = np.array(mass_diff_puppi_)
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='green', linewidth=linewidth, 
             density=True, label=r'PUPPI, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    PUPPIMassdiffMu, PUPPIMassSigma = getStat(mass_diff)
    mass_diff = np.array(mass_diff_puppi_wcut_)
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='red', linewidth=linewidth, 
             density=True, label=r'PF, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    mass_diff = np.array(mass_diff_CHS_)
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='orange', linewidth=linewidth, 
             density=True, label=r'CHS, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    # plt.xlim(-1.0,1.3)
    plt.xlabel(r"Jet Mass $(m_{reco} - m_{truth})/m_{truth}$")
    plt.ylabel('density')
    plt.ylim(0, 6)
    plt.rc('legend', fontsize=fontsize)
    
    plt.legend()
    plt.savefig(args.save_dir+"/prob_plots/Jet_mass_diff_"+postfix+str(epoch)+".pdf")
    plt.close()
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    neutral_weight_total = np.array(neu_pred)
    chlv_weight_total = np.array(chlv_pred)
    chpu_weight_total = np.array(chpu_pred)
    plt.hist(neutral_weight_total, bins=400, range=(0, 1.05), histtype='step', color='blue', linewidth=linewidth,
              density=False, label=r'Neutral particle weight')
    plt.hist(chlv_weight_total, bins=400, range=(0, 1.05), histtype='step', color='green', linewidth=linewidth,
              density=False, label=r'Charged LV particle weight')
    plt.hist(chpu_weight_total, bins=400, range=(0, 1.05), histtype='step', color='pink', linewidth=linewidth,
              density=False, label=r'Charged PU particle weight')
        
    plt.xlabel(r"SSL weight")
    plt.ylabel('A.U.')
    plt.legend()
    plt.savefig(args.save_dir+"/prob_plots/GNNweight_"+postfix+str(epoch)+".pdf")
    plt.close()
    plt.show()
    

    

    return total_loss,total_loss_puppi,total_loss_pf, SSLMassdiffMu, \
        SSLMassSigma, PUPPIMassdiffMu, PUPPIMassSigma


def generate_jet_mass_truth(dataset):
    # calculate Genjet inv mass for each graph
    for graph in dataset:
        pt_truth = np.array(graph.GenPart_nump[:, 2].cpu().detach())
        eta_truth = np.array(graph.GenPart_nump[:, 0].cpu().detach())
        phi_truth = np.array(graph.GenPart_nump[:, 1].cpu().detach())
        graph.JetmassTruth = GetJetInvMass(pt_truth, eta_truth, phi_truth)
    
    return dataset

def main():
    args = arg_parse()
    print("model type: ", args.model_type)

    # load the constructed graphs
    with open(args.training_path, "rb") as fp:
        dataset = pickle.load(fp)
    with open(args.validation_path, "rb") as fp:
        dataset_validation = pickle.load(fp)

    generate_jet_mass_truth(dataset)
    generate_jet_mass_truth(dataset_validation)
    train(dataset, dataset_validation, args, 1)


if __name__ == '__main__':
    main()
