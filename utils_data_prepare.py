import os
import os.path as osp
import argparse
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import torch.nn.functional as F


from k_gnn import TwoMalkin, ConnectedThreeMalkin, ConnectedThreeLocal
from TDGNN_utils import *

def prepare_dataset_master(args, device):
    if '1_2_3_GNN' in args.GNN_model:
        dataset = k_GNN_create_dataset(args)
    else:
        dataset = DropGIN_create_dataset(args, device)
    return dataset

### loading the dataset for DropGIN
def DropGIN_create_dataset(args, device):
    ### download the standard TU dataset
    ### define the filter and the transformation of the dataset
    ### set "path" to the path of our dataset
    ### load the dataset into "dataset"
    if 'IMDB' in args.dataset: #IMDB-BINARY or #IMDB-MULTI
        class MyFilter(object):
            def __call__(self, data):
                return data.num_nodes <= 70

        class MyPreTransform(object):
            def __call__(self, data):
                data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                data.x = F.one_hot(data.x, num_classes=136).to(torch.float)#136 in k-gnn?
                return data

        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')

        dataset = TUDataset(path, name=args.dataset, pre_transform=MyPreTransform(), pre_filter=MyFilter())

    elif 'MUTAG' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True
        
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')

        dataset = TUDataset(path, name=args.dataset, pre_filter=MyFilter())
    elif 'PROTEINS' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return not (data.num_nodes == 7 and data.num_edges == 12) and data.num_nodes < 450

        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')
        
        dataset = TUDataset(path, name=args.dataset, pre_filter=MyFilter())
    elif 'PTC' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True        
        
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')
        
        dataset = TUDataset(path, name=args.dataset, pre_filter=MyFilter())
    else:
        raise ValueError
    
    if 'TDGNN' in args.GNN_model:
        ### define the save path for the addtional information for TDGNN
        save_path = os.path.join(path, 'TDGNN')

        ### check whether the save directory exists
        if(osp.exists(save_path) == False):
            os.mkdir(save_path)
            for i in tqdm(range(len(dataset)), desc = 'creating the hop_edge_index and hop_edge_att'):
                #run tree decomposition
                data = dataset[i]
                
                ### universal
                edge_index_location = os.path.join(save_path, f'hop_edge_index_{args.dataset}_{args.tree_layer}_{i}.pt')
                edge_att_location = os.path.join(save_path, f'hop_edge_att_{args.dataset}_{args.tree_layer}_{i}.pt') 

                data.edge_index_location = edge_index_location
                data.edge_att_location = edge_att_location
                edge_info(data, args)
            print('Processing finished!')

        i = 0
        ### loading the hop_edge_index and hop_edge_att
        for i in tqdm(range(0, len(dataset)), desc = 'loading the hop_edge_index and hop_edge_att'):
            
            ### universal
            hop_edge_index = torch.load(os.path.join(save_path, f'hop_edge_index_{args.dataset}_{args.tree_layer}_{i}.pt'))
            hop_edge_att = torch.load(os.path.join(save_path, f'hop_edge_att_{args.dataset}_{args.tree_layer}_{i}.pt'))

            for layer in args.layers:
                hop_edge_index[layer - 1] = hop_edge_index[layer - 1].type(torch.LongTensor).to(device)
                hop_edge_att[layer - 1] = hop_edge_att[layer - 1].to(device)
            setattr(dataset[i].__class__, "hop_edge_index", hop_edge_index)
            setattr(dataset[i].__class__, "hop_edge_att", hop_edge_att)
        print('Loading finished!')

    return dataset


### loading the dataset for k_GNN
def k_GNN_create_dataset(args):
    if 'PROTEINS' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return not (data.num_nodes == 7 and data.num_edges == 12) and data.num_nodes < 450

        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'1-2-3-{args.dataset}')

        dataset = TUDataset(path, name=args.dataset, 
                            pre_transform=T.Compose([TwoMalkin(), ConnectedThreeMalkin()]), 
                            pre_filter=MyFilter())
    elif 'MUTAG' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True

        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'1-2-3-{args.dataset}')

        dataset = TUDataset(path, name=args.dataset, 
                            pre_transform=T.Compose([TwoMalkin(), ConnectedThreeLocal()]), 
                            pre_filter=MyFilter())

    elif 'IMDB' in args.dataset: #IMDB-BINARY or #IMDB-MULTI
        class MyFilter(object):
            def __call__(self, data):
                return data.num_nodes <= 70

        class MyPreTransform(object):
            def __call__(self, data):
                data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
                data = TwoMalkin()(data)
                data = ConnectedThreeMalkin()(data)
                data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                data.x = F.one_hot(data.x, num_classes=136).to(torch.float)
                return data

        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'1-2-3-{args.dataset}')

        dataset = TUDataset(path, name=args.dataset, 
                            pre_transform=MyPreTransform(), 
                            pre_filter=MyFilter())

    elif 'PTC' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                # return data.x.shape[0] > 2
                return True
        
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'1-2-3-{args.dataset}')
        
        dataset = TUDataset(path, name=args.dataset, 
                            pre_filter=MyFilter(), 
                            pre_transform=T.Compose([TwoMalkin(), ConnectedThreeMalkin()]))
    
    else:
         raise ValueError

    dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
    num_i_2 = dataset.data.iso_type_2.max().item() + 1
    dataset.data.iso_type_2 = F.one_hot(dataset.data.iso_type_2,
                                        num_classes=num_i_2).to(torch.float)

    dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
    num_i_3 = dataset.data.iso_type_3.max().item() + 1
    dataset.data.iso_type_3 = F.one_hot(dataset.data.iso_type_3,
                                        num_classes=num_i_3).to(torch.float)
    dataset.data.num_i_2 = num_i_2
    dataset.data.num_i_3 = num_i_3
    
    ### somehow the first element of PTC is causing problems for k-GNN
    dataset = dataset[1:]
    return dataset
    

### returning a list:[list_1, ..., list_{n_splits - 1}]
### list_0 = [list_a = [0, 1, 2, ..., {dataset_len - 1}], list_b]
### len(list_b) = int(len(dataset)/n_splits)
def separate_data(dataset_len, seed=0):
        # Use same splitting/10-fold as GIN paper
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
        idx_list = []
        for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
            idx_list.append(idx)
        return idx_list

