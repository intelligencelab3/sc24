import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from dataset_prep import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler

import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_dense_adj

from logger import Logger
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import time
import copy
import subprocess
#%matplotlib inline
#import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from partition_util.partition_cluster import *
from torch_sparse import SparseTensor

#torch.set_num_threads(80)
torch.manual_seed(999)
# 1042, 0, 999

class SAGE_MULT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE_MULT, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # two linear layer for predictions
        self.linear = torch.nn.ModuleList()
        self.linear.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        
        self.bn0 = BatchNorm1d(hidden_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.linear:
            lin.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            
        # print(x[0])
        x = self.linear[0](x)
        x = self.bn0(F.relu(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = self.linear[1](x) # for xor
        x2 = self.linear[2](x) # for maj
        x3 = self.linear[3](x) # for roots
        # print(self.linear[0].weight)
        # print(x1[0])
        return x, x1.log_softmax(dim=-1), x2.log_softmax(dim=-1), x3.log_softmax(dim=-1)
    def forward_nosampler(self, x, adj_t, device):
        # tensor placement
        x.to(device)
        adj_t.to(device)
        
        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # print(x[0])
        x = self.linear[0](x)
        x = self.bn0(F.relu(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = self.linear[1](x) # for xor
        x2 = self.linear[2](x) # for maj
        x3 = self.linear[3](x) # for roots
        # print(self.linear[0].weight)
        # print(x1[0])
        return x1, x2, x3

    def forward_nosampler_partitioned(self, x, adj_t, device):
        # tensor placement
        # x.to(device)
        # adj_t.to(device)
        
        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # print(x[0])
        x = self.linear[0](x)
        x = self.bn0(F.relu(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = self.linear[1](x) # for xor
        x2 = self.linear[2](x) # for maj
        x3 = self.linear[3](x) # for roots
        # print(self.linear[0].weight)
        # print(x1[0])
        return x1, x2, x3

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                xs.append(x)

                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
            #print(x_all.size())
            
        x_all = self.linear[0](x_all)
        x_all = F.relu(x_all)
        x_all = self.bn0(x_all)
        x1 = self.linear[1](x_all) # for xor
        x2 = self.linear[2](x_all) # for maj
        x3 = self.linear[3](x_all) # for roots
        pbar.close()

        return x1, x2, x3  
     
def train(model, data_r, data, train_idx, optimizer, train_loader, device):
    pbar = tqdm(total=train_idx.size(0))

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        _, out1, out2, out3 = model(data.x[n_id], adjs)
        
        ### build labels for multitask
        ### original 0: PO, 1: plain, 2: shared, 3: maj, 4: xor, 5: PI
        y1 = data.y.squeeze(1)[n_id[:batch_size]].clone().detach() # make (maj and xor) as xor
        for i in range(y1.size()[-1]):
            if y1[i] == 0 or y1[i] == 5:
                y1[i] = 1
            if y1[i] == 2: 
                y1[i] = 4
            if y1[i] > 2:
                y1[i] = y1[i] - 1 # make to 5 classes
            y1[i] = y1[i] - 1 # 3 classes: 0: plain, 1: maj, 2: xor
                
        y2 = data.y.squeeze(1)[n_id[:batch_size]].clone().detach() # make (maj and xor) as maj
        for i in range(y2.size()[-1]):
            if y2[i] > 2:
                y2[i] = y2[i] - 1 # make to 5 classes
            if y2[i] == 0 or y2[i] == 4:
                y2[i] = 1
            y2[i] = y2[i] - 1 # 3 classes: 0: plain, 1: maj, 2: xor
                
        # for root classification
        # 0: PO, 1: maj, 2: xor, 3: and, 4: PI
        # y3 = data_r.y.squeeze(1)[n_id[:batch_size]]
        y3 = data_r.y.squeeze(1)[n_id[:batch_size]].clone().detach()
        for i in range(y3.size()[-1]):
            if y3[i] == 0 or y3[i] == 4:
                y3[i] = 3
            y3[i] = y3[i] - 1 # 3 classes: 0: maj, 1: xor, 2: and+PI+PO

            
        loss =  F.nll_loss(out1, y1) + F.nll_loss(out2, y2) + 0.8 * F.nll_loss(out3, y3)
        
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out1.argmax(dim=-1).eq(y1).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc

def post_processing(out1, out2):
    pred_1 = out1.argmax(dim=-1, keepdim=True)
    pred_2 = out2.argmax(dim=-1, keepdim=True)
    pred_ecc = (out1 + out2).argmax(dim=-1, keepdim=True)
    # l =  pred_1.size()[0]
    # pred = []
    # for i in range(l):
    #     if pred_1[i] == pred_2[i]:
    #         if pred_1[i] == 0: # PO, and, PI
    #             pred.append(torch.tensor([1]))
    #         else: # maj, xor
    #             pred.append(pred_1[i] + 2) # 3 or 4
    #     else:
    #         if (pred_1[i] == 1 and pred_2[i] == 2) or (pred_1[i] == 2 and pred_2[i] == 1):
    #             pred.append(torch.tensor([2])) # maj and xor
    #         else:
    #             if pred_ecc[i] == 0: # PO, and, PI
    #                 pred.append(torch.tensor([1]))
    #             else: # maj, xor
    #                 pred.append(pred_ecc[i] + 2)
    # pred = torch.tensor(pred)
    
    pred = copy.deepcopy(pred_1)
    
    eq_idx = (torch.eq(pred_1, pred_2) == True).nonzero(as_tuple=True)[0]
    # if pred_1[i] != 0  # maj, xor
    eq_mx_idx = (pred_1[eq_idx] != 0).nonzero(as_tuple=True)[0]
    # pred_1[i] = pred_1[i] + 2  -->  3, 4
    pred[eq_idx[eq_mx_idx]] = pred_1[eq_idx[eq_mx_idx]] + 2
    # if pred_1[i] == 0 PI/PI/and --> final 1
    eq_aig_idx = (pred_1[eq_idx] == 0).nonzero(as_tuple=True)[0]
    pred[eq_idx[eq_aig_idx]] = 1

    neq_idx = (torch.eq(pred_1, pred_2) == False).nonzero(as_tuple=True)[0]
    # if pred_1[i] == 1 and pred_2[i] == 2 shared --> 2
    p1 = (pred_1[neq_idx] == 1).nonzero(as_tuple=True)[0]
    p2 = (pred_2[neq_idx] == 2).nonzero(as_tuple=True)[0]
    shared = p1[(p1.view(1, -1) == p2.view(-1, 1)).any(dim=0)]
    pred[neq_idx[shared]] = 2
    # else (error correction for discrepant predictions)
    if len(p1) != len(p2) or len(p1) != len(neq_idx):
        v, freq = torch.unique(torch.cat((p1, p2), 0), sorted=True, return_inverse=False, return_counts=True, dim=None)
        uniq = (freq == 1).nonzero(as_tuple=True)[0]
        ecc = v[uniq]
        ecc_mx = (pred_ecc[neq_idx][ecc] != 0).nonzero(as_tuple=True)[0]
        ecc_aig = (pred_ecc[neq_idx][ecc] == 0).nonzero(as_tuple=True)[0]
        pred[neq_idx[ecc[ecc_mx]]] = pred_ecc[neq_idx][ecc][ecc_mx] + 2
        pred[neq_idx[ecc[ecc_aig]]] = 1

    return torch.reshape(pred, (pred.shape[0], 1))  
       

@torch.no_grad()
def test(model, data_r, data, split_idx, evaluator, subgraph_loader, datatype, device):
    model.eval()

    start_time = time.time()
    out1, out2, out3 = model.inference(data.x, subgraph_loader, device)
    y_pred_shared = post_processing(out1, out2)
    y_pred_root = out3.argmax(dim=-1, keepdim=True)
    # print("print output stats of model.inference", out1.shape, out2.shape)
    print('The inference time is %s' % (time.time() - start_time))
    y_shared = data.y.squeeze(1).clone().detach()
    y_root = data_r.y.squeeze(1).clone().detach()
    
    
    # for i in range(y_shared.size()[-1]): # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
    #     if y_shared[i] == 0 or y_shared[i] == 5:
    #         y_shared[i] = 1
    # for i in range(y_root.size()[-1]):
    #     if y_root[i] == 0 or y_root[i] == 4:
    #         y_root[i] = 3
    #     y_root[i] = y_root[i] - 1
        
    # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
    s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
    s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
    y_shared[s5] = 1
    y_shared[s0] = 1
    
    r0 = (y_root == 0).nonzero(as_tuple=True)[0]
    r4 = (y_root == 4).nonzero(as_tuple=True)[0]
    y_root[r0] = 3
    y_root[r4] = 3
    y_root = y_root - 1
    
    y_root = torch.reshape(y_root, (y_root.shape[0], 1))
    y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))  
    
    
    # print(y_pred_root.size())
    # print(y_pred_shared.size())
    
    # print(y_root.size())
    # print(y_shared.size())
    
    if datatype=='train':
        train_acc_r = evaluator.eval({
            'y_true': y_root[split_idx['train']],
            'y_pred': y_pred_root[split_idx['train']],
        })['acc']
        valid_acc_r = evaluator.eval({
            'y_true': y_root[split_idx['valid']],
            'y_pred': y_pred_root[split_idx['valid']],
        })['acc']
        test_acc_r = evaluator.eval({
            'y_true': y_root[split_idx['test']],
            'y_pred': y_pred_root[split_idx['test']],
        })['acc']
        train_acc_s = evaluator.eval({
            'y_true': y_shared[split_idx['train']],
            'y_pred': y_pred_shared[split_idx['train']],
        })['acc']
        valid_acc_s = evaluator.eval({
            'y_true': y_shared[split_idx['valid']],
            'y_pred': y_pred_shared[split_idx['valid']],
        })['acc']
        test_acc_s = evaluator.eval({
            'y_true': y_shared[split_idx['test']],
            'y_pred': y_pred_shared[split_idx['test']],
        })['acc']
        # print("print output label shape", data.y[split_idx['test']].shape)
        return train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s
    else:
        test_acc_r = evaluator.eval({
            'y_true': y_root,
            'y_pred': y_pred_root,
        })['acc']
        test_acc_s = evaluator.eval({
            'y_true': y_shared,
            'y_pred': y_pred_shared
        })['acc']

        return 0, 0, test_acc_r, 0, 0, test_acc_s
    
@torch.no_grad()
def test_nosampler_partitioned(model, data_r, data, num_partitions, recovery, datasetname, split_idx, evaluator, datatype, device):
    model.eval()
    ####### Added code for partition #######
    adj_t_coo = data.adj_t.coo() 
    data.edge_index = torch.stack((adj_t_coo[0], adj_t_coo[1]), dim = 0)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)

    train_mask[split_idx['train']] = True
    val_mask[split_idx['valid']] = True
    test_mask[split_idx['test']] = True
    data.train_mask, data.test_mask, data.val_mask = train_mask, test_mask, val_mask
    data_partitioned = partition_graph(data.cpu(), num_partitions, dataset = datasetname)
    data_partitioned = data_partitioned.to(device)
    out1 = torch.zeros([data.num_nodes, 3])
    out1 = out1.to(device)
    out2 = torch.zeros([data.num_nodes, 3])
    out2 = out2.to(device)
    out3 = torch.zeros([data.num_nodes, 3])
    out3 = out3.to(device)
    edge_index_all = []
    edge_index_inner = []
    for i in range(num_partitions):
        cluster_info = data_partitioned.cluster_nodes_edges[i]
        edge_index = SparseTensor(row=cluster_info['all_edges'][0], col=cluster_info['all_edges'][1], sparse_sizes=(data.num_nodes, data.num_nodes))
        edge_index_all.append(edge_index)
        edge_index = SparseTensor(row=cluster_info['inner_edges'][0], col=cluster_info['inner_edges'][1], sparse_sizes=(data.num_nodes, data.num_nodes))
        edge_index_inner.append(edge_index)
    if recovery: 
        print("Inference with boundary recovery")
    else:
        print("Inference without boundary recovery")


    # Reset the peak memory statistics
    torch.cuda.reset_peak_memory_stats(device)
    # peak_memory_start = torch.cuda.max_memory_allocated(device)
    for i in range (5):
        start_time = time.time()
        if recovery: 
            for i in range(num_partitions):
                cluster_info = data_partitioned.cluster_nodes_edges[i]
            # edge_index = SparseTensor(row=cluster_info['all_edges'][0], col=cluster_info['all_edges'][1], sparse_sizes=(data.num_nodes, data.num_nodes))
                out1_temp, out2_temp, out3_temp\
                     = model.forward_nosampler_partitioned(data_partitioned.x, edge_index_all[i], device)
            #torch.stack((cluster_info['all_edges'][0], cluster_info['all_edges'][1]), dim = 0)
                out1[cluster_info['inner_nodes_id']], out2[cluster_info['inner_nodes_id']], out3[cluster_info['inner_nodes_id']]\
                    = out1_temp[cluster_info['inner_nodes_id']], out2_temp[cluster_info['inner_nodes_id']], out3_temp[cluster_info['inner_nodes_id']]
        else:
            for i in range(num_partitions):
                cluster_info = data_partitioned.cluster_nodes_edges[i]
            # edge_index = SparseTensor(row=cluster_info['inner_edges'][0], col=cluster_info['inner_edges'][1], sparse_sizes=(data.num_nodes, data.num_nodes))
                out1_temp, out2_temp, out3_temp\
                      = model.forward_nosampler_partitioned(data_partitioned.x, edge_index_inner[i], device)
            #torch.stack((cluster_info['inner_edges'][0], cluster_info['inner_edges'][1]), dim = 0)
                out1[cluster_info['inner_nodes_id']], out2[cluster_info['inner_nodes_id']], out3[cluster_info['inner_nodes_id']]\
                      = out1_temp[cluster_info['inner_nodes_id']], out2_temp[cluster_info['inner_nodes_id']], out3_temp[cluster_info['inner_nodes_id']]
    # edge_index = SparseTensor(row=data_partitioned.edge_index[0], col=data_partitioned.edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes))
    # start_time = time.time()
    # out1, out2, out3\
    #         = model.forward_nosampler_partitioned(data_partitioned.x, edge_index, device)
    # data = data.to(device)
    # out1, out2, out3\
    #         = model.forward_nosampler_partitioned(data.x, torch.stack((adj_t_coo[0], adj_t_coo[1]), dim = 0), device)

    ####### Added code for partition --End #######
    # start_time = time.time()
    # out1, out2, out3 = model.forward_nosampler(data.x, data.adj_t, device)

        y_pred_shared = post_processing(out1, out2)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)
        print('The inference time is %s' % (time.time() - start_time))
    
#################################### Warm up code #################################################################

    for i in range (5):
        start_time = time.time()
        if recovery: 
            for i in range(num_partitions):
                cluster_info = data_partitioned.cluster_nodes_edges[i]
            # edge_index = SparseTensor(row=cluster_info['all_edges'][0], col=cluster_info['all_edges'][1], sparse_sizes=(data.num_nodes, data.num_nodes))
                out1_temp, out2_temp, out3_temp\
                     = model.forward_nosampler_partitioned(data_partitioned.x, edge_index_all[i], device)
            #torch.stack((cluster_info['all_edges'][0], cluster_info['all_edges'][1]), dim = 0)
                out1[cluster_info['inner_nodes_id']], out2[cluster_info['inner_nodes_id']], out3[cluster_info['inner_nodes_id']]\
                    = out1_temp[cluster_info['inner_nodes_id']], out2_temp[cluster_info['inner_nodes_id']], out3_temp[cluster_info['inner_nodes_id']]
        else:
            for i in range(num_partitions):
                cluster_info = data_partitioned.cluster_nodes_edges[i]
            # edge_index = SparseTensor(row=cluster_info['inner_edges'][0], col=cluster_info['inner_edges'][1], sparse_sizes=(data.num_nodes, data.num_nodes))
                out1_temp, out2_temp, out3_temp\
                      = model.forward_nosampler_partitioned(data_partitioned.x, edge_index_inner[i], device)
            #torch.stack((cluster_info['inner_edges'][0], cluster_info['inner_edges'][1]), dim = 0)
                out1[cluster_info['inner_nodes_id']], out2[cluster_info['inner_nodes_id']], out3[cluster_info['inner_nodes_id']]\
                      = out1_temp[cluster_info['inner_nodes_id']], out2_temp[cluster_info['inner_nodes_id']], out3_temp[cluster_info['inner_nodes_id']]
    # edge_index = SparseTensor(row=data_partitioned.edge_index[0], col=data_partitioned.edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes))
    # start_time = time.time()
    # out1, out2, out3\
    #         = model.forward_nosampler_partitioned(data_partitioned.x, edge_index, device)
    # data = data.to(device)
    # out1, out2, out3\
    #         = model.forward_nosampler_partitioned(data.x, torch.stack((adj_t_coo[0], adj_t_coo[1]), dim = 0), device)

    ####### Added code for partition --End #######
    # start_time = time.time()
    # out1, out2, out3 = model.forward_nosampler(data.x, data.adj_t, device)

        y_pred_shared = post_processing(out1, out2)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)
        print('The inference time is %s' % (time.time() - start_time))


# Get the peak memory usage
    peak_memory_end = torch.cuda.max_memory_allocated(device)
    peak_memory = peak_memory_end #- peak_memory_start
    print('The peak GPU memory consumption is {} bytes, or {} MB'.format(peak_memory, peak_memory/1024/1024))

    # kiran modified for the different memory stats
    memory_allocated=torch.cuda.memory_allocated()
    print('The GPU memory allocated is {} bytes, or {} MB'.format(memory_allocated, memory_allocated/1024/1024))
    
    memory_reserved=torch.cuda.memory_reserved()
    print('The GPU memory reserved is {} bytes, or {} MB'.format(memory_reserved, memory_reserved/1024/1024))

    memory_reserved_max=torch.cuda.max_memory_reserved()
    print('The GPU memory max reserved is {} bytes, or {} MB'.format(memory_reserved_max, memory_reserved_max/1024/1024))
    

    result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"])
    print('nvidia-smi output:')
    print(result.decode('utf-8'))
    







    ####### Added code for partition #######
    data_partitioned =data_partitioned.cpu()
    y_pred_shared = y_pred_shared[data_partitioned.reverse_map]
    y_pred_root = y_pred_root[data_partitioned.reverse_map]
    ####### Added code for partition --End #######
    print("print output stats of model.inference", out1.shape, out2.shape)   #kiran modeified
    # tensor placement
    y_shared = data.y.squeeze(1).clone().detach().to(device)
    y_root = data_r.y.squeeze(1).clone().detach().to(device)
    
    # for i in range(y_shared.size()[-1]): # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
    #     if y_shared[i] == 0 or y_shared[i] == 5:
    #         y_shared[i] = 1
    # for i in range(y_root.size()[-1]):
    #     if y_root[i] == 0 or y_root[i] == 4:
    #         y_root[i] = 3
    #     y_root[i] = y_root[i] - 1
    
    s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
    s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
    y_shared[s5] = 1
    y_shared[s0] = 1
    
    r0 = (y_root == 0).nonzero(as_tuple=True)[0]
    r4 = (y_root == 4).nonzero(as_tuple=True)[0]
    y_root[r0] = 3
    y_root[r4] = 3
    y_root = y_root - 1
    #print(out3.shape)
    #print(f'Dataset: {out3}:')
    
    y_root = torch.reshape(y_root, (y_root.shape[0], 1))
    y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))  
    #print(f'Dataset: {y_root}:')
    print('===================================================================================================')
   # print(f'Number of graphs: {len(data_r)}')
    #print(f'Number of features: {data_r.num_features}')
    #print(f'Number of classes: {data_r.num_classes}')

    
    # print(y_pred_root.size())
    # print(y_pred_shared.size())
    
    # print(y_root.size())
    # print(y_shared.size())

    
    if datatype=='train':
        train_acc_r = evaluator.eval({
            'y_true': y_root[split_idx['train']],
            'y_pred': y_pred_root[split_idx['train']],
        })['acc']
        valid_acc_r = evaluator.eval({
            'y_true': y_root[split_idx['valid']],
            'y_pred': y_pred_root[split_idx['valid']],
        })['acc']
        test_acc_r = evaluator.eval({
            'y_true': y_root[split_idx['test']],
            'y_pred': y_pred_root[split_idx['test']],
        })['acc']
        train_acc_s = evaluator.eval({
            'y_true': y_shared[split_idx['train']],
            'y_pred': y_pred_shared[split_idx['train']],
        })['acc']
        valid_acc_s = evaluator.eval({
            'y_true': y_shared[split_idx['valid']],
            'y_pred': y_pred_shared[split_idx['valid']],
        })['acc']
        test_acc_s = evaluator.eval({
            'y_true': y_shared[split_idx['test']],
            'y_pred': y_pred_shared[split_idx['test']],
        })['acc']
        # print("print output label shape", data.y[split_idx['test']].shape)
        return train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s
    else:
        test_acc_r = evaluator.eval({
            'y_true': y_root,
            'y_pred': y_pred_root,
        })['acc']
        test_acc_s = evaluator.eval({
            'y_true': y_shared,
            'y_pred': y_pred_shared
        })['acc']

        return 0, 0, test_acc_r, 0, 0, test_acc_s



def confusion_matrix_plot(model, data_r, data, subgraph_loader, device, datatype, save_file, bit_train, bit_test):
    model.eval()

    out1, out2, out3 = model.inference(data.x, subgraph_loader, device)
    # print("print output stats of model.inference", out1.shape, out2.shape)
    y_pred_shared = post_processing(out1, out2)
    y_pred_root = out3.argmax(dim=-1, keepdim=True)
    
    y_shared = data.y.squeeze(1).clone().detach()
    y_root = data_r.y.squeeze(1).clone().detach()
    
    for i in range(y_shared.size()[-1]): # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
        if y_shared[i] == 0 or y_shared[i] == 5:
            y_shared[i] = 1
    for i in range(y_root.size()[-1]):
        if y_root[i] == 0 or y_root[i] == 4:
            y_root[i] = 3
        y_root[i] = y_root[i] - 1

    if save_file == True:
        #pd.DataFrame(y_pred[split_idx[datatype]].numpy()).to_csv('pred_' + str(datatype) + '.csv')
        #pd.DataFrame(data.y[split_idx[datatype]].numpy()).to_csv('label_' + str(datatype) + '.csv')
        pd.DataFrame(y_pred_root.numpy()).to_csv('pred_root_plain_' + str(datatype) + '_' + str(bit_train) + 'to' + str(bit_test) + '.csv', index = False, header = False)
        pd.DataFrame(y_root.numpy()).to_csv('label_root_plain_' + str(datatype) + '_' + str(bit_train) + 'to' + str(bit_test) + '.csv', index = False, header = False)
        pd.DataFrame(y_pred_shared.numpy()).to_csv('pred_shared_plain_' + str(datatype) + '_' + str(bit_train) + 'to' + str(bit_test) + '.csv', index = False, header = False)
        pd.DataFrame(y_shared.numpy()).to_csv('label_shared_plain_' + str(datatype) + '_' + str(bit_train) + 'to' + str(bit_test) + '.csv', index = False, header = False)
        
    # plot confusion matrix for shared nodes
    conf_matrix = confusion_matrix(y_shared.numpy(), y_pred_shared.numpy())
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greys)
    ax.set_xticklabels(['', '0', '1', '2', '3'], fontsize=22)
    ax.set_yticklabels(['', '0', '1', '2', '3'], fontsize=22)
    font = {'size':22}

    plt.rc('font', **font)
    plt.xlabel('Predictions', fontsize=22)
    plt.ylabel('Actuals', fontsize=22)
    #plt.title('Confusion Matrix', fontsize=25)
    plt.savefig('confusion_matrix_plain_shared_' + str(datatype) + '_' + str(bit_train) + 'to' + str(bit_test) + '.pdf', bbox_inches = 'tight',pad_inches = 0)
    
    # plot confusion matrix for roots
    conf_matrix = confusion_matrix(y_root.numpy(), y_pred_root.numpy())
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greys)
    ax.set_xticklabels(['', '0', '1', '2'], fontsize=22)
    ax.set_yticklabels(['', '0', '1', '2'], fontsize=22)
    font = {'size':22}

    plt.rc('font', **font)
    plt.xlabel('Predictions', fontsize=22)
    plt.ylabel('Actuals', fontsize=22)
    #plt.title('Confusion Matrix', fontsize=25)
    plt.savefig('confusion_matrix_plain_root_' + str(datatype) + '_' + str(bit_train) + 'to' + str(bit_test) + '.pdf', bbox_inches = 'tight',pad_inches = 0)
  
  
  
def write_txt(model, data, subgraph_loader, device, file_name, num_class):
    model.eval()
    out1, out2, out3 = model.inference(data.x, subgraph_loader, device)
    
    y_pred_shared = post_processing(out1, out2)
    y_pred_root = out3.argmax(dim=-1, keepdim=True)
    
    # y_pred = post_processing(out1, out2, out3)
    # print(y_pred)
    
    y_pred_list = y_pred_shared.flatten().numpy()
    f = open(file_name + '_plain_shared.txt', 'w')
    for i in range(num_class):
        line = np.where(y_pred_list == i)[0]
        # print(line)
        f.write(','.join(str(ind) for ind in line))
        f.write('\n')
    f.close()
    
    y_pred_list = y_pred_root.flatten().numpy()
    f = open(file_name + '_plain_root.txt', 'w')
    for i in range(5):
        line = np.where(y_pred_list == i)[0]
        # print(line)
        f.write(','.join(str(ind) for ind in line))
        f.write('\n')
    f.close()
      

def main():
    parser = argparse.ArgumentParser(description='mult16')
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--bits_test', type=int, default=32)
    parser.add_argument('--task', type=str, choices = ['csa', 'booth'], default='csa')
    parser.add_argument('--datagen', type=int, default=0,
		help="0=multiplier generator, 1=adder generator, 2=loading design")
    # (0)(1) require bits as inputs; (2) requires designfile as input
    parser.add_argument('--datagen_test', type=int, default=0,
		help="0=multiplier generator, 1=adder generator, 2=loading design")
    # (0)(1) require bits as inputs; (2) requires designfile as input
    parser.add_argument('--multilabel', type=int, default=1,
        help="0=5 classes; 1=6 classes with shared xor/maj as a new class; 2=multihot representation")
    parser.add_argument('--num_class', type=int, default=6)
    parser.add_argument('--designfile', '-f', type=str, default='')
    parser.add_argument('--designfile_test', '-ft', type=str, default='')
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--design_copies', type=int, default=1)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--mapped', type=int, default=0)
    parser.add_argument('--recovery', action='store_true', default=False)
    parser.add_argument('--num-partitions', type=int, default=4)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = torch.device('cpu') ## cpu for now only

    ## generate dataset
    root_folder = os.path.abspath((os.path.dirname(os.path.abspath("gnn_sampler.py"))))
    print(root_folder)
    # new generator functionalities: 0) multiplier 1) adder 2) read design

    if args.task == "booth":
        prefix = 'booth_mult'
        suffix_m = ''
    elif args.task == "csa":
        if args.datagen == 0:
            prefix = 'mult'
    design_name = prefix + str(args.bits)
    
    ### training dataset loading
    dataset_r = PygNodePropPredDataset(name = design_name + '_root', task = args.task)
    print("Training on %s" % design_name)
    data_r = dataset_r[0]
#######kiran_modifications########################################################################

    #def visualize_graph(G, color):
     ##  plt.yticks([])
        #nx.draw_networkx(G,pos=nx.spring_layout(G, seed=42),with_labels=True,
        #             node_color=color, )
       # nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
     #                node_color=color, cmap="Set2")
                     
        #plt.show()


    #G = to_networkx(data_r, to_undirected=True)
    #visualize_graph(G, color=data_r.y)
    #bb = nx.edge_betweenness_centrality(G, normalized=False)
    # print(bb)
    # G_new=nx.set_edge_attributes(G, bb, "betweenness")
    #visualize_graph(G_new, color=data_r.y)
    #print(list(G.nodes))
    #cut_value, partition = nx.minimum_cut(G, 0, 17,capacity=1)
    #print(cut_value)
    #print(G.shape)
   # A = nx.adjacency_matrix(G)
 #   print(A)

     #A=to_dense_adj(data_r.edge_index)
   # print(A.shape)
   #print(A)


# Helper function for visualization.


    #def visualize(h, color):
     #   z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

     #   plt.figure(figsize=(10,10 ))
    #    plt.xticks([])
    #    plt.yticks([])
#
    #    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    #    plt.show()

   # G = to_networkx(data_r, to_undirected=True)
    #visualize(G, color=data_r.y)
  
    print(f'Dataset: {data_r}:')
    print('===================================================================================================')
    print(f'Number of graphs: {len(data_r)}')
    print(f'Number of features: {data_r.num_features}')
    #print(f'Number nodes: {data_r.nodes}')
    #print(f'Number of classes: {data_r.num_classes}')

    #A=to_dense_adj(data_r.edge_index)
    #print(A.shape)
    #print(A)
    print('===================================================================================================')

    data_r = T.ToSparseTensor()(data_r)
    
    dataset = PygNodePropPredDataset(name = design_name + '_shared', task = args.task)
    data = dataset[0]
    G_shared = to_networkx(data, to_undirected=False)
    # visualize_graph(G_shared, color=data_r.y)
    print(f'Dataset: {data}:')
    print('================================================================================================')
    print(f'Number of graphs: {len(data)}')
    print(f'Number of features: {data.num_features}')



    data = T.ToSparseTensor()(data)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    train_loader = NeighborSampler(data.adj_t, node_idx=train_idx,
                               sizes=[8, 5, 5, 3], batch_size = 20,
                               shuffle=True)
    subgraph_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  )
    
    evaluator = Evaluator(name = design_name + '_shared')

    model = SAGE_MULT(data.num_features, args.hidden_channels,
                     3, args.num_layers,
                     args.dropout).to(device)
    
    data_r = data_r.to(device)
    data = data.to(device)

    logger_r = Logger(args.runs, args)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = 5e-5)
        for epoch in range(1, 1 + args.epochs):
            loss, _ = train(model, data_r, data, train_idx, optimizer, train_loader, device)
            # train val
            datatype='train'
            result = test(model, data_r, data, split_idx, evaluator, subgraph_loader, datatype, device)
            logger_r.add_result(run, result[:3])
            logger.add_result(run, result[3:])

            if epoch % args.log_steps == 0:
                train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s= result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'[Root Model] Train: {100 * train_acc_r:.2f}%, '
                      f'[Root Model] Valid: {100 * valid_acc_r:.2f}% '
                      f'[Root Model] Test: {100 * test_acc_r:.2f}% '
                      f'[Shared Model] Train: {100 * train_acc_s:.2f}%, '
                      f'[Shared Model] Valid: {100 * valid_acc_s:.2f}% '
                      f'[Shared Model] Test: {100 * test_acc_s:.2f}%')
                
        logger_r.print_statistics(run)
        logger.print_statistics(run)
    logger_r.print_statistics()
    logger.print_statistics()
    
    if args.save_model:
      #  model_name = 'SAGE_mult' + str(args.bits) Kiran modifying to get the model for the batch processing
        model_name = 'SAGE_mult' + str(args.bits)
        torch.save(model.state_dict(), model_name)
    
    ### evaluation dataset loading
    logger_eval_r = Logger(1, args)
    logger_eval = Logger(1, args)
    if args.task == "booth":
        prefix = 'booth_mult'
        suffix = ''
    elif args.task == "csa":
        if args.datagen_test == 0:
            prefix = 'mult'
        elif args.datagen_test == 1:
            prefix = 'adder'
        if args.mapped == 1:
            suffix ="_7nm_mapped"
        else:
            suffix = ''
    #if args.datagen_test == 0:
     #   prefix = 'mult'
    design_name = prefix + str(args.bits_test) + suffix
    #design_name = prefix + str(args.bits_test)
    print("Evaluation on %s" % design_name)
    
    dataset_r = PygNodePropPredDataset(name = design_name + '_root')
    data_r = dataset_r[0]
    data_r = T.ToSparseTensor()(data_r)
    
    dataset = PygNodePropPredDataset(name = design_name + '_shared')
    data = dataset[0]
    data = T.ToSparseTensor()(data)
    split_idx = dataset.get_idx_split()
    subgraph_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1],
                                  batch_size=40960000, shuffle=False,
                                  )
    
    # tensor placement
    data_r = data_r.to(device)
    data = data.to(device)
    #print(f'root_Dataset: {data_r}:')
    #G = to_networkx(data_r, to_undirected=True)
    #visualize_graph(G, color=data_r.y)
    evaluator = Evaluator(name = design_name + '_shared')
   
    for run_1 in range(1):
        for epoch in range(1):
            datatype='test'
            # This one uses neighbor sampler
            #result = test(model, data_r, data, split_idx, evaluator, subgraph_loader, datatype, device)
            result = test_nosampler_partitioned(model, data_r, data, args.num_partitions, args.recovery, design_name + '_shared', split_idx, evaluator, datatype, device)
            #print(f'Dataset: {result}:')
            #print(result.shape)
            logger_eval_r.add_result(run_1, result[:3])
            logger_eval.add_result(run_1, result[3:])
            if epoch % args.log_steps == 0:
                train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s = result
                print(f'Run: {run_1 + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'[Root Model] Train: {100 * train_acc_r:.2f}%, '
                      f'[Root Model] Valid: {100 * valid_acc_r:.2f}% '
                      f'[Root Model] Test: {100 * test_acc_r:.2f}% '
                      f'[Shared Model] Train: {100 * train_acc_s:.2f}%, '
                      f'[Shared Model] Valid: {100 * valid_acc_s:.2f}% '
                      f'[Shared Model] Test: {100 * test_acc_s:.2f}%')
                
        # logger_eval_r.print_statistics(run_1)
        # logger_eval.print_statistics(run_1)
        
    #confusion_matrix_plot(model, data_r, data, subgraph_loader, device, datatype='test', save_file=True, bit_train = args.bits, bit_test = args.bits_test)
    
    # file_name = 'pred_test_rowclass'
    # write_txt(model, data, subgraph_loader, device, file_name, args.num_class)
    logger_eval_r.print_statistics()
    logger_eval.print_statistics()
    
    
    
    #model.eval()

    #out = model(data.x, data.edge_index)
    #visualize(out, color=data.y)
    



if __name__ == "__main__":
    main()