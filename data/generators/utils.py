import os

import numpy as np
import torch

import torch_geometric.datasets as datasets
import torch_geometric.transforms as T
# from networkx import from_scipy_sparse_matrix

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix

from ogb.nodeproppred import PygNodePropPredDataset
import scipy.io as sio
import pickle as pkl
import sys
import scipy.sparse as sp
import networkx as nx
def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)
    torch.save(data, fpath)
def get_mat_data(file_path):

    
    data = sio.loadmat(file_path)
    x = torch.tensor(data['Attributes'].todense(), dtype=torch.float)
    y = torch.tensor(data['Class'], dtype=torch.long).squeeze()
    label = torch.tensor(data['Label'], dtype=torch.float).squeeze()
    
    edge_index = torch.tensor(np.vstack((data['Network'].nonzero())), dtype=torch.long)
    
    num_nodes = x.size(0)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

       
    return Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, label=label)


def split_train(data, dataset, data_path, ratio_train, mode, n_clients):
    n_data = data.num_nodes
    ratio_test = (1 - ratio_train) / 2
    n_train = round(n_data * ratio_train)
    n_test = round(n_data * ratio_test)

    permuted_indices = torch.randperm(n_data)
    train_indices = permuted_indices[:n_train]
    test_indices = permuted_indices[n_train:n_train + n_test]
    val_indices = permuted_indices[n_train + n_test:]

    data.train_mask.fill_(False)
    data.test_mask.fill_(False)
    data.val_mask.fill_(False)

    data.train_mask[train_indices] = True
    data.test_mask[test_indices] = True
    data.val_mask[val_indices] = True

    torch_save(data_path, f'{dataset}_{mode}/{n_clients}/train.pt', {'data': data})
    torch_save(data_path, f'{dataset}_{mode}/{n_clients}/test.pt', {'data': data})
    torch_save(data_path, f'{dataset}_{mode}/{n_clients}/val.pt', {'data': data})
    print(f'splition done, n_train: {n_train}, n_test: {n_test}, n_val: {len(val_indices)}')
    return data

def get_data(dataset, data_path):
    
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        
        data = datasets.Planetoid(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
    elif dataset in ['Computers', 'Photo']:
        data = datasets.Amazon(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
       
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in ['ogbn-arxiv']:
        data = PygNodePropPredDataset(dataset, root=data_path, transform=T.Compose([T.ToUndirected(), LargestConnectedComponents()]))[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
        data.y = data.y.view(-1)
    return data

class LargestConnectedComponents(BaseTransform):
    
    def __init__(self, num_components: int = 1):
        self.num_components = num_components

    def __call__(self, data: Data) -> Data:
        import numpy as np
        import scipy.sparse as sp

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(adj)

        if num_components <= self.num_components:
            return data

        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-self.num_components:])

        return data.subgraph(torch.from_numpy(subset).to(torch.bool))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_components})'

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
dataset_str = 'cora'
def get_raw_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("raw_dataset/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("raw_dataset/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj_dense = np.array(adj.todense(), dtype=np.float64)
    attribute_dense = np.array(features.todense(), dtype=np.float64)
    cat_labels = np.array(np.argmax(labels, axis = 1).reshape(-1,1), dtype=np.uint8)
    adj_coo = adj.tocoo()
    edge_index = np.vstack((adj_coo.row, adj_coo.col))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    y = torch.tensor(cat_labels, dtype=torch.long)
    new_data = Data(x=attribute_dense, edge_index=edge_index,adj=adj, y=cat_labels)
    
    
    return new_data