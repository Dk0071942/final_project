import os
import os.path as osp
from matplotlib.style import use
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
# from test_tube import HyperOptArgumentParser
# from test_tube.hpc import SlurmCluster
import argparse
from tqdm import tqdm
import json
import sys

import torch
from torch import dropout, nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import uniform
import torch_geometric.transforms as T

from GNN_models import GIN, DropGIN, Net, GNN_VN, TDGNN
# from models_graph_classification import GNNSubstructures
from TDGNN_utils import *
from k_gnn import DataLoader as k_DataLoader, GraphConv, avg_pool
from k_gnn import TwoMalkin, ConnectedThreeMalkin, ConnectedThreeLocal
from utils_data_prepare import prepare_dataset_master, separate_data


def train(model, epoch, loader, optimizer, device, use_aux_loss):
        loss_all = 0

        for data in loader:
            ### DataBatch(edge_index=[2, 3216], x=[846, 3], y=[32], batch=[846], ptr=[33])
            data = data.to(device)
            optimizer.zero_grad()

            if model.__class__.__name__ == 'GNN_VN':
                logs = model(data)
                print(logs)            
                loss = torch.nn.BCEWithLogitsLoss()(logs.to(torch.float32), torch.reshape(data.y, (32, 1)).to(torch.float32))
            else:
                logs, aux_logs = model(data)
                # print(logs)
                ### logs.shape ---> torch.Size([32, 2])
                ### data.y ---> torch.size([0])
                loss = F.nll_loss(logs, data.y)
                if use_aux_loss:
                    aux_loss = F.nll_loss(aux_logs.view(-1, aux_logs.size(-1)), data.y.unsqueeze(0).expand(aux_logs.size(0),-1).clone().view(-1))
                    loss = 0.75*loss + 0.25*aux_loss

            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
        
        return loss_all / len(loader.dataset)


def val(model, loader, device):
    model.eval()
    with torch.no_grad():
        loss_all = 0
        for data in loader:
            data = data.to(device)
            ### for bianry-classification only
            if model.__class__.__name__ == 'GNN_VN':
                logs = model(data)
                loss = torch.nn.BCEWithLogitsLoss()(logs.to(torch.float32), torch.reshape(data.y, (data.y, 1)).to(torch.float32))
                loss_all += loss
            else:
                logs, aux_logs = model(data)
                loss_all += F.nll_loss(logs, data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)


def test(model, loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        for data in loader:
            data = data.to(device)
            ### not working!!!!!
            if model.__class__.__name__ == 'GNN_VN':
                logs = model(data)
                loss = torch.nn.BCEWithLogitsLoss()(logs.to(torch.float32), torch.reshape(data.y, (data.y.shape[0], 1)).to(torch.float32))
            else:
                logs, aux_logs = model(data)
                pred = logs.max(1)[1]
                correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)        


def load_gnn_model(dataset, args, device):
    use_aux_loss = False    
    if '1_2_3_GNN' in args.GNN_model:
        model = Net(dataset, args)
    elif 'DropGIN' == args.GNN_model:
        model = DropGIN(dataset, args)
        use_aux_loss = args.use_aux_loss
    elif 'GIN' == args.GNN_model:
        model = GIN(dataset, args)
    elif 'GCN_virtual' == args.GNN_model:
        model = GNN_VN(1, args, 'gcn', True)
    elif 'GIN_virtual' == args.GNN_model:
        model = GNN_VN(1, args, 'gin', True)
    elif 'TDGNN' == args.GNN_model:
        model = TDGNN(dataset, args)
    # elif 'GSN' == args.GNN_model:
    #     model == GNNSubstructures(dataset,args)
    model = model.to(device)
    
    return model, use_aux_loss

def add_underscores():
    dash_len = len('----------------------------------------------------------------------------------------------------')
    print('-' * dash_len) 

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("result.txt", "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

def main(args):

    sys.stdout = Logger()

    add_underscores()
    print("Starting new training")

    run_info = vars(args)
    for key, value in run_info.items():
        print(key, ':', value)

    BATCH = args.batch_size

    ### set up the device
    torch.manual_seed(0)
    np.random.seed(0)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dataset = prepare_dataset_master(args, device)

    ### select the model
    model, use_aux_loss = load_gnn_model(dataset, args, device)

    acc = []
    splits = separate_data(len(dataset), seed=0)
    
    print('Current GNN model: ' + model.__class__.__name__)
    print('Current dataset: ' + dataset.name)
    
    for i, (train_idx, test_idx) in enumerate(splits):
        model.reset_parameters()
        lr = args.lr
        ### set the optimizer 
        if args.GNN_model == 'TDGNNN':
            if(args.agg == 'sum'):
                optimizer = torch.optim.Adam([
                        dict(params=model.lin1.parameters(), weight_decay=args.wd1),
                        dict(params=model.lin2.parameters(), weight_decay=args.wd2)], lr=args.lr)
            elif(args.agg == 'weighted_sum'):
                optimizer = torch.optim.Adam([
                        dict(params=model.lin1.parameters(), weight_decay=args.wd1),
                        dict(params=model.lin2.parameters(), weight_decay=args.wd2),
                        dict(params=model.prop.parameters(), weight_decay=args.wd3)], lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) # in GIN code 50 itters per epoch were used
        test_dataset = dataset[test_idx.tolist()]
        train_dataset = dataset[train_idx.tolist()]
        

        if '1_2_3_GNN' in args.GNN_model:
            test_loader = k_DataLoader(test_dataset, batch_size=BATCH)
            train_loader = k_DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
        else:
            test_loader = DataLoader(test_dataset, batch_size=BATCH)
            train_loader = torch.utils.data.DataLoader(train_dataset, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(len(train_dataset)*50/(len(train_dataset)/BATCH))), batch_size=BATCH, drop_last=False, collate_fn=Collater(follow_batch=[],exclude_keys=[]))	# GIN like epochs/batches - they do 50 radom batches per epoch


        print('---------------- Split {} ----------------'.format(i), flush=True)

        test_acc = 0
        acc_temp = []
        for epoch in range(1, args.epochs+1):
            if args.verbose or epoch == args.epochs:
                start = time.time()
            lr = scheduler.optimizer.param_groups[0]['lr']
            train_loss = train(model, epoch, train_loader, optimizer, device, use_aux_loss)
            scheduler.step()
            test_acc = test(model, test_loader, device)
            if args.verbose or epoch == args.epochs:
                print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                    'Val Loss: {:.7f}, Test Acc: {:.7f}, Time: {:7f}'.format(
                        epoch, lr, train_loss, 0, test_acc, time.time() - start), flush=True)
            acc_temp.append(test_acc)
        acc.append(torch.tensor(acc_temp))
    acc = torch.stack(acc, dim=0)
    acc_mean = acc.mean(dim=0)
    best_epoch = acc_mean.argmax().item()
    print('---------------- Final Epoch Result ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc[:,-1].mean(), acc[:,-1].std()))
    print(f'---------------- Best Epoch: {best_epoch} ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc[:,best_epoch].mean(), acc[:,best_epoch].std()), flush=True)

    add_underscores()

    run_info['final_epoch_mean'] = acc[:,-1].mean()
    run_info['final_epoch_std'] = acc[:,-1].std()
    run_info['best_epoch_mean'] = acc[:,best_epoch].mean()
    run_info['best_epoch_std'] = acc[:,best_epoch].std()
    print(run_info)

    add_underscores()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### options=[0.5, 0.0]
    parser.add_argument('--dropout', type=float, default=0.5)
    ### options=[32, 128]
    parser.add_argument('--batch_size', type=int, default=32)
    # 64 is used for social datasets (IMDB) and 16 or 32 for bio datasest (MUTAG, PTC, PROTEINS). 
    parser.add_argument('--hidden_units', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_layers', type=int, default=4, help='number of GNN message passing layers (default: 4)')    
    parser.add_argument('--use_aux_loss', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='MUTAG', help="Options are ['MUTAG', 'PTC_MR', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI']")
    parser.add_argument('--GNN_model', type=str, default='1_2_3_GNN', help="Options are ['GIN', 'DropGIN', '1_2_3_GNN', 'TDGNN', 'GIN_virtual', 'GCN_virtual']")
    args = parser.parse_args()
    ### TDGNN
    if args.GNN_model == 'TDGNN':
        parser.add_argument('--wd1', type=float, default=0.006)
        parser.add_argument('--wd2', type=float, default=0.006)
        parser.add_argument('--wd3', type=float, default=0)
        parser.add_argument('--early_stopping', type=int, default=0)
        parser.add_argument('--dropout1', type=float, default=0.8)
        parser.add_argument('--dropout2', type=float, default=0.8)
        parser.add_argument('--normalize_features', type=bool, default=True)
        parser.add_argument('--K_TDGNN', type=int, default=10)
        parser.add_argument('--tree_layer', type=int, default=10)
        parser.add_argument('--layers', nargs='+', type = int, default=(1,2,3,4))
        parser.add_argument('--setting', type=str, default='semi')
        parser.add_argument('--agg', type=str, default='sum')
        parser.add_argument('--tree_decompose', type=bool, default=True)
    elif 'virtual' in args.GNN_model:
    ### GNN_Virtual_node
        parser.add_argument('--emb_dim', type=int, default=300)

    args = parser.parse_args()

    main(args)

    print('Finished', flush=True)