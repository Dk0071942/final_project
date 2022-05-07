import torch
from torch import dropout, nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GINConv
from torch_geometric.nn.inits import uniform
from virtual_node_conv import GNN_node, GNN_node_Virtualnode
import torch_geometric.transforms as T
from k_gnn import DataLoader as k_DataLoader, GraphConv, avg_pool
from k_gnn import TwoMalkin, ConnectedThreeMalkin, ConnectedThreeLocal
from torch_scatter import scatter_mean


### Class GIN
class GIN(nn.Module):
    def __init__(self, dataset, args):
        super(GIN, self).__init__()

        num_features = dataset.num_features
        num_classes = dataset.num_classes
        dim = args.hidden_units
        
        self.dropout = args.dropout

        self.num_layers = args.num_layers

        ### list convs
        self.convs = nn.ModuleList()
        ### list bns
        self.bns = nn.ModuleList()
        ### list fcs
        self.fcs = nn.ModuleList()

        self.convs.append(GINConv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
        ### Applies Batch Normalization over a 2D or 3D input
        self.bns.append(nn.BatchNorm1d(dim))
        ### Applies a linear transformation to the incoming data
        self.fcs.append(nn.Linear(num_features, num_classes))
        self.fcs.append(nn.Linear(dim, num_classes))

        for i in range(self.num_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, num_classes))
    
    ### reset the parameters for those layers
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        outs = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            ### here the new x has the same shape as data.x
            outs.append(x)

        ### outs=[data.x, new_x_0, ... , new_x_num_layers - 1]

        out = None
        for i, x in enumerate(outs):
            ### the new x below this line has the shape [batch_size, num_features]             
            x = global_add_pool(x, batch)
            ### the new x below this line has the shape [batch_size, num_classes]
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x
            ### out has the shape [batch_size, num_classes]  
        return F.log_softmax(out, dim=-1), 0


### Class DropGIN
class DropGIN(nn.Module):
    def __init__(self, dataset, args):    
        super(DropGIN, self).__init__()

        use_aux_loss = args.use_aux_loss
        self.use_aux_loss = use_aux_loss
        
        num_features = dataset.num_features
        dim = args.hidden_units
        self.dropout = args.dropout

        self.num_layers = args.num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(GINConv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(num_features, dataset.num_classes))
        self.fcs.append(nn.Linear(dim, dataset.num_classes))

        for i in range(self.num_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))
        
        if use_aux_loss:
            self.aux_fcs = nn.ModuleList()
            self.aux_fcs.append(nn.Linear(num_features, dataset.num_classes))
            for i in range(self.num_layers):
                self.aux_fcs.append(nn.Linear(dim, dataset.num_classes))
                
        # Set the sampling probability and number of runs/samples for the DropGIN
        n = []
        degs = []
        for g in dataset:
            num_nodes = g.num_nodes
            deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
            n.append(g.num_nodes)
            degs.append(deg.max())
            
        print('-----------------------------------------------------------------------')
        print('Set the sampling probability and number of runs/samples for the DropGIN')    
        print(f'Mean Degree: {torch.stack(degs).float().mean()}')
        print(f'Max Degree: {torch.stack(degs).max()}')
        print(f'Min Degree: {torch.stack(degs).min()}')
        mean_n = torch.tensor(n).float().mean().round().long().item()
        print(f'Mean number of nodes: {mean_n}')
        print(f'Max number of nodes: {torch.tensor(n).float().max().round().long().item()}')
        print(f'Min number of nodes: {torch.tensor(n).float().min().round().long().item()}')
        print(f'Number of graphs: {len(dataset)}')
        gamma = mean_n
        p = 2 * 1 /(1+gamma)
        num_runs = gamma
        print(f'Number of runs: {num_runs}')
        print(f'Sampling probability: {p}')
        print('-----------------------------------------------------------------------')
        self.num_runs = num_runs
        self.p = p
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        num_runs = self.num_runs
        p = self.p
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        # Do runs in paralel, by repeating the graphs in the batch
        x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device)*p).bool()
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
        del drop
        outs = [x]
        x = x.view(-1, x.size(-1))
        run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
        for i in range(self.num_layers):
            x = self.convs[i](x, run_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x.view(num_runs, -1, x.size(-1)))
        del  run_edge_index
        out = None
        for i, x in enumerate(outs):
            x = x.mean(dim=0)
            x = global_add_pool(x, batch)
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x

        use_aux_loss = self.use_aux_loss


        if use_aux_loss:
            aux_out = torch.zeros(num_runs, out.size(0), out.size(1), device=out.device)
            run_batch = batch.repeat(num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(batch.size(0)) * (batch.max() + 1)
            for i, x in enumerate(outs):
                x = x.view(-1, x.size(-1))
                x = global_add_pool(x, run_batch)
                x = x.view(num_runs, -1, x.size(-1))
                x = F.dropout(self.aux_fcs[i](x), p=self.dropout, training=self.training)
                aux_out += x

            return F.log_softmax(out, dim=-1), F.log_softmax(aux_out, dim=-1)
        else:
            return F.log_softmax(out, dim=-1), 0
 
 
###  Class 1-2-3-GNN      
class Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(Net, self).__init__()
        num_i_2 =  dataset.data.num_i_2
        num_i_3 = dataset.data.num_i_3
        self.conv1 = GraphConv(dataset.num_features, 32)
        self.conv2 = GraphConv(32, 64)
        self.conv3 = GraphConv(64, 64)
        self.conv4 = GraphConv(64 + num_i_2, 64)
        self.conv5 = GraphConv(64, 64)
        self.conv6 = GraphConv(64 + num_i_3, 64)
        self.conv7 = GraphConv(64, 64)
        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, dataset.num_classes)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        data.x = F.elu(self.conv3(data.x, data.edge_index))
        x = data.x
        x_1 = scatter_mean(data.x, data.batch, dim=0)


        data.x = avg_pool(x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2))
        x_2 = scatter_mean(data.x, data.batch_2, dim=0)

        data.x = avg_pool(x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        # if args.no_train:
        #     x = x.detach()

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), 0


###  Class GNN_VN      
class GNN_VN(torch.nn.Module):

    ### default
    ### emb_dim = 300 
    ### gnn_type = 'gin'
    ### virtual_node = True
    def __init__(self, num_tasks, args, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN_VN, self).__init__()

        self.num_layer = args.num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = args.emb_dim        
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(self.num_layer, self.emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = 'gin')
        else:
            self.gnn_node = GNN_node(self.num_layer, self.emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(self.emb_dim, 2*self.emb_dim), torch.nn.BatchNorm1d(2*self.emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*self.emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(self.emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
            # self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
            # self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)


    def reset_parameters(self):
        pass

    def forward(self, batched_data):
        ### h_node has the shape [batched_data.num_nodes, emb_dim]
        h_node = self.gnn_node(batched_data)

        ### h_graph has the shape [batch_size, emb_dim]
        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)


### Class TDGNN prop_sum
class prop_sum(MessagePassing):
    def __init__(self, num_classes, layers, **kwargs):
        super(prop_sum, self).__init__(aggr = 'add', **kwargs)
        self.layers = layers

    def forward(self, x, edge_index, edge_weight):
        embed_layer = []
        embed_layer.append(x)

        if(self.layers != [0]):
            for layer in self.layers:
                # edge_weight[layer - 1] = edge_weight[layer - 1]/torch.sum(edge_weight[layer - 1])
                h = self.propagate(edge_index[layer - 1], x = x, norm = edge_weight[layer - 1])
                embed_layer.append(h)

        embed_layer = torch.stack(embed_layer, dim = 1)

        return embed_layer

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        pass


### Class TDGNN prop_weight
class prop_weight(MessagePassing):
    def __init__(self, num_classes, layers, **kwargs):
        super(prop_weight, self).__init__(aggr = 'add', **kwargs)

        self.weight = torch.nn.Parameter(torch.ones(len(layers) + 1), requires_grad = True)

        self.layers = layers

    def forward(self, x, edge_index, edge_weight):
        embed_layer = []
        embed_layer.append(self.weight[0] * x)

        for i in range(len(self.layers)):
            h = self.propagate(edge_index[self.layers[i] - 1], x = x, norm = edge_weight[self.layers[i] - 1])
            embed_layer.append(self.weight[i + 1] * h)


        embed_layer = torch.stack(embed_layer, dim = 1)

        return embed_layer

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(torch.ones(len(self.layers) + 1), requires_grad = True)


### Class TDGNN
class TDGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(TDGNN, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.agg = args.agg
        self.num_classes = dataset.num_classes

        self.lin1 = Linear(dataset.num_features, self.args.hidden_units)
        self.lin2 = Linear(self.args.hidden_units, dataset.num_features)
        self.lin3 = Linear(dataset.num_features, self.num_classes)

        if(self.agg == 'sum'):
            self.prop = prop_sum(dataset.num_classes, self.args.layers)
        if(self.agg == 'weighted_sum'):
            self.prop = prop_weight(dataset.num_classes, self.args.layers)


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        x = F.dropout(x, p = self.args.dropout1, training = self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p = self.args.dropout2, training = self.training)
        x = self.lin2(x)

        # x = self.prop(x, self.args.hop_edge_index, self.args.hop_edge_att)
        x = self.prop(x, data.hop_edge_index, data.hop_edge_att)
        ### the x below has the same shape as data.x
        x = torch.sum(x, dim = 1)
        ### the x below has the shape [batch_size, num_features]
        x = global_add_pool(x, batch)

        ### out has the shape [batch_size, num_classes]
        out = F.dropout(self.lin3(x), p=self.dropout, training=self.training)

        return F.log_softmax(x, dim = 1), 0

        