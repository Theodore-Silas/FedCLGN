import random

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix

from utils import torch_load
import torch
import networkx as nx

class DataLoader:
    def __init__(self, args):
        self.args = args
        self.n_workers = 1
        self.client_id = None

        from torch_geometric.loader import DataLoader
        self.DataLoader = DataLoader

    def switch(self, client_id):
        if not self.client_id == client_id:
            self.client_id = client_id
            if self.args.dataset ==  'Flickr':
                self.partition = load_pt1(get_data(self.args, client_id=client_id))
            else:
                self.partition = load_pt(get_data(self.args, client_id=client_id))
            self.pa_loader = self.DataLoader(dataset=self.partition, batch_size=1,
                                             shuffle=False, num_workers=self.n_workers, pin_memory=False)
            self.pa_loader = self.DataLoader(dataset=self.partition, batch_size=1,
                                             shuffle=False, num_workers=self.n_workers, pin_memory=False)


def get_data(args, client_id):
    return [
        torch_load(
            args.data_path,
            f'{args.dataset}_{args.mode}/{args.n_clients}/partition_{client_id}.pt'
        )['client_data']
    ]


class GraphData:
    def __init__(self, adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels, node_indices, client_dict):
        self.adj = adj
        self.feat = feat
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.ano_labels = ano_labels
        self.str_ano_labels = str_ano_labels
        self.attr_ano_labels = attr_ano_labels
        self.len = 1
        self.node_indices = node_indices
        self.client_dict = client_dict

    def __repr__(self):
        return f"GraphData(adj={self.adj.shape}, feat={self.feat.shape}, labels={self.labels.shape}, " \
               f"idx_train={len(self.idx_train)}, idx_val={len(self.idx_val)}, idx_test={len(self.idx_test)}, " \
               f"ano_labels={self.ano_labels.shape}, str_ano_labels={self.str_ano_labels.shape}, " \
               f"attr_ano_labels={self.attr_ano_labels.shape}, len={self.len})"


def dense_to_one_hot(labels_dense, num_classes):
    
    #Convert class labels from scalars to one-hot vectors

    num_labels = labels_dense.shape[0]  
    index_offset = np.arange(num_labels) * num_classes  
    labels_one_hot = np.zeros((num_labels, num_classes))  
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1  
    return labels_one_hot


def load_pt(data, train_rate=0.3, val_rate=0.1):
    
    label = data[0].label
    attr = data[0].x
    network = data[0].edge_index

    
    num_nodes = attr.shape[0]
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.int32)
    adj[network[0], network[1]] = 1
    adj[network[1], network[0]] = 1

    
    attr = csc_matrix(attr)

    adj = sp.csr_matrix(adj)
    feat = sp.lil_matrix(attr)

    labels = np.squeeze(np.array(data[0].y, dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels, num_classes)  

    ano_labels = np.squeeze(np.array(label))

    str_ano_labels = None
    attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]  #
    node_indices = data[0].node_indices
    client_dict = data[0].client_dict
    
    return GraphData(adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels,
                     node_indices, client_dict)



def load_pt1(data, train_rate=0.3, val_rate=0.1):
    label = data[0].label
    attr = data[0].x
    network = data[0].edge_index

    # Create a NetworkX graph from the edge index
    G = nx.Graph()
    edges = list(zip(network[0].tolist(), network[1].tolist()))  # Ensure edges are lists
    G.add_edges_from(edges)

    # Find the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc = sorted(largest_cc)  # Sort to maintain a consistent order

    # Create a mapping from original node IDs to indices in the largest connected component
    node_map = {node: i for i, node in enumerate(largest_cc)}

    # Filter nodes and edges based on the largest connected component
    filtered_edges = [(node_map[u], node_map[v]) for u, v in edges if u in node_map and v in node_map]
    attr_filtered = attr[largest_cc, :]
    label_filtered = label[largest_cc]

    # Initialize the adjacency matrix
    num_nodes = len(largest_cc)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.int32)

    for u, v in filtered_edges:
        adj[u, v] = 1
        adj[v, u] = 1  # Ensure symmetry for undirected graph

    # Convert matrices to sparse formats
    adj = sp.csr_matrix(adj)
    feat = sp.lil_matrix(attr_filtered)

    # Prepare labels
    labels = np.squeeze(np.array(label_filtered, dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels, num_classes)

    ano_labels = np.squeeze(np.array(label_filtered))

    str_ano_labels = None
    attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    node_indices = [node_map[node] for node in data[0].node_indices if node in node_map]
    client_dict = {node_map[key]: value for key, value in data[0].client_dict.items() if key in node_map}

    return GraphData(adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels,
                     node_indices, client_dict)