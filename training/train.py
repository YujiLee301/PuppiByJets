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
import test_physics_metrics as phym
import matplotlib
from copy import deepcopy
import os
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
                        num_layers=3,
                        batch_size=4,
                        hidden_dim=20,
                        dropout=0,
                        opt='adam',
                        weight_decay=0,
                        lr=0.001,
                        pulevel=80,
                        training_path="../data_pickle/dataset_graph_puppi_8000",
                        validation_path="../data_pickle/dataset_graph_puppi_val_4000",
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
        px, py, px, ene = SetPxPyPzE(pt[i], eta[i], phi[i])
        Px+=px
        Py+=py
        Pz+=Pz
        Energy+=ene
    
    return math.sqrt(Energy*Energy-Px*Px-Py*Py-Pz*Pz)


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
        dataset[0].num_feature_actual, args.hidden_dim, 1, args)
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

    train_graph_SSLMassdiffMu = []
    train_graph_PUPPIMassdiffMu = []
    train_graph_SSLMassSigma = []
    train_graph_PUPPIMassSigma = []
    valid_graph_SSLMassdiffMu = []
    valid_graph_PUPPIMassdiffMu = []
    valid_graph_SSLMassSigma = []
    valid_graph_PUPPIMassSigma = []

    count_event = 0
    converge = False
    converge_num_event = 0
    last_steady_event = 0
    lowest_valid_loss = 10

    while converge == False:
        model.train()

        t = tqdm(total=len(training_loader), colour='green',
                 bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        loss_avg = utils.RunningAverage()
        for batch in training_loader:
            count_event += 1
            epochs_train.append(count_event)
            batch = batch.to(device)
            pt = np.array(batch.x[:, 2])
            eta = np.array(batch.x[:, 0])
            phi = np.array(batch.x[:, 1])
            pred, _ = model.forward(batch)
            loss = model.loss(GetJetInvMass(pt*pred,eta,phi),batch.JetmassTruth)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_graph.append(loss)
            # print("cur_loss ", cur_loss)
            loss_avg.update(loss)
            # print("loss_avg ", loss_avg())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

            if count_event % 500 == 0:

                modelcolls = OrderedDict()
                modelcolls['gated_boost'] = model
                training_loss,train_SSLMassdiffMu, \
                    train_SSLMassSigma, train_PUPPIMassdiffMu, train_PUPPIMassSigma = test(
                        training_loader, model, 0, count_event, args, modelcolls, args.training_path)

                valid_loss, valid_SSLMassdiffMu, \
                    valid_SSLMassSigma, valid_PUPPIMassdiffMu, valid_PUPPIMassSigma = test(
                        validation_loader, model, 1, count_event, args, modelcolls, args.validation_path)

                epochs_valid.append(count_event)
                loss_graph_valid.append(valid_loss)
                loss_graph_train.append(training_loss)

                train_graph_SSLMassdiffMu.append(train_SSLMassdiffMu)
                train_graph_PUPPIMassdiffMu.append(train_PUPPIMassdiffMu)
                train_graph_SSLMassSigma.append(train_SSLMassSigma)
                train_graph_PUPPIMassSigma.append(train_PUPPIMassSigma)

                valid_graph_SSLMassdiffMu.append(valid_SSLMassdiffMu)
                valid_graph_PUPPIMassdiffMu.append(valid_PUPPIMassdiffMu)
                valid_graph_SSLMassSigma.append(valid_SSLMassSigma)
                valid_graph_PUPPIMassSigma.append(valid_PUPPIMassSigma)

                if (valid_SSLMassSigma/(1-abs(valid_SSLMassdiffMu))) < (best_validation_SSLMassSigma/(1-abs(best_valid_SSLMassdiffMu))):
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

                if count_event == 5000:
                    converge = True
                    break

        t.close()

    end = timer()
    training_time = end - start
    print("training time " + str(training_time))

    #To-do plotting loss here
    plt.figure()
    plt.plot(epochs_valid, loss_graph_valid, label = 'valid', linestyle = 'solid', linewidth = 1, color = 'b')
    plt.plot(epochs_valid, loss_graph_train, label = 'train', linestyle = 'solid', linewidth = 1, color = 'g')
    #plt.plot(epochs_valid, valid_graph_PUPPIMassdiffMu, label = 'PUPPI_valid_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.savefig(args.save_dir + "/loss.pdf")
    plt.close()

    plt.figure()
    plt.plot(epochs_valid, train_graph_SSLMassdiffMu, label = 'Semi-supervised_train_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_valid, valid_graph_SSLMassdiffMu, label = 'Semi-supervised_valid_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'b')
    plt.plot(epochs_valid, train_graph_PUPPIMassdiffMu, label = 'PUPPI_train_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'r')
    #plt.plot(epochs_valid, valid_graph_PUPPIMassdiffMu, label = 'PUPPI_valid_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('mean diff')
    plt.legend(loc=4)
    plt.savefig(args.save_dir + "/Jet_mass_diff_mean.pdf")
    plt.close()

    plt.figure()
    plt.plot(epochs_valid, train_graph_SSLMassSigma, label = 'Semi-supervised_train_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_valid, valid_graph_SSLMassSigma, label = 'Semi-supervised_valid_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'b')
    plt.plot(epochs_valid, train_graph_PUPPIMassSigma, label = 'PUPPI_train_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'r')
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
    count = 0
    for data in loader:
        count += 1
        if count == epoch and indicator == 0:
            break
        with torch.no_grad():
            num_feature = data.num_feature_actual[0].item()
            test_mask = data.x[:, num_feature]

            data.x = torch.cat(
                (data.x[:, 0:num_feature], test_mask.view(-1, 1), data.x[:, -num_feature:]), 1)
            data = data.to(device)
            # max(dim=1) returns values, indices tuple; only need indices
            pred, pred_hybrid = model.forward(data)
            # puppi = data.x[:, data.num_feature_actual[0].item() - 1]
            puppi = data.pWeight
            label = data.y

            if pred_all != None:
                pred_all = torch.cat((pred_all, pred), 0)
                pred_hybrid_all = torch.cat((pred_hybrid_all, pred_hybrid), 0)
                puppi_all = torch.cat((puppi_all, puppi), 0)
                label_all = torch.cat((label_all, label), 0)
            else:
                pred_all = pred
                pred_hybrid_all = pred_hybrid
                puppi_all = puppi
                label_all = label

            mask_neu = data.mask_neu[:, 0]

            if test_mask_all != None:
                test_mask_all = torch.cat((test_mask_all, test_mask), 0)
                mask_all_neu = torch.cat((mask_all_neu, mask_neu), 0)
            else:
                test_mask_all = test_mask
                mask_all_neu = mask_neu

            label = label[test_mask == 1]
            pred = pred[test_mask == 1]
            pred_hybrid = pred_hybrid[test_mask == 1]
            label = label.type(torch.float)
            label = label.view(-1, 1)
            total_loss += model.loss(pred, label).item() * data.num_graphs

    if indicator == 0:
        total_loss /= min(epoch, len(loader.dataset))
    else:
        total_loss /= len(loader.dataset)
    
    filelists = []
    filelists.append(pathname)

    mass_diff_CHS, mass_diff_puppi, mass_diff_puppi_wcut, mass_diff_pred, neu_weight, neu_puppiweight, chlv_weight, chpu_weight, chlv_puppiweight, chpu_puppiweight  =phym.test(
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
    mass_diff = np.array(mass_diff_pred)
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='blue', linewidth=linewidth,
             density=True, label=r'Semi-supervised, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    SSLMassdiffMu, SSLMassSigma = getStat(mass_diff)
    mass_diff = np.array(mass_diff_puppi)
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='green', linewidth=linewidth, 
             density=True, label=r'PUPPI, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    PUPPIMassdiffMu, PUPPIMassSigma = getStat(mass_diff)
    mass_diff = np.array(mass_diff_puppi_wcut)
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='red', linewidth=linewidth, 
             density=True, label=r'PF, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    mass_diff = np.array(mass_diff_CHS)
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='orange', linewidth=linewidth, 
             density=True, label=r'CHS, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    # plt.xlim(-1.0,1.3)
    plt.xlabel(r"Jet Mass $(m_{reco} - m_{truth})/m_{truth}$")
    plt.ylabel('density')
    plt.ylim(0, 6)
    plt.rc('legend', fontsize=fontsize)
    
    plt.legend()
    plt.savefig(args.save_dir+"/prob_plots/Jet_mass_diff_"+postfix+str(epoch)+".pdf")
    plt.show()

    

    return total_loss, SSLMassdiffMu, \
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