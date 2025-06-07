import torch
import random
import numpy as np
import scipy.io as sio

import metispy as metis

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils import  split_train, torch_save, get_mat_data,get_data, get_raw_data
from scipy.spatial.distance import euclidean
data_path = '../../datasets'

ratio_train = 0.2
seed = 1234
clients = [5, 10, 15]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
AD_dataset_list = ['BlogCatalog', 'Flickr']
Citation_dataset_list = ['cora', 'citeseer', 'pubmed']
def generate_data(dataset, n_clients):
    
    dataset_str  = 'cora' # cora citeseer BlogCatalog
    if dataset_str in AD_dataset_list:
        data = sio.loadmat('./raw_dataset/{}/{}.mat'.format(dataset_str, dataset_str))
        attribute_dense = np.array(data['Attributes'].todense())
        adj_dense = np.array(data['Network'].todense())
        edge_index = torch.tensor(np.vstack((data['Network'].nonzero())), dtype=torch.long)

        cat_labels = data['Label']

        data = Data(x=attribute_dense, edge_index=edge_index, y=cat_labels)
    elif dataset_str in Citation_dataset_list:
        data = get_raw_data(dataset_str)
    split_subgraphs(n_clients, data, dataset)

def add_anomaly_nodes(data, m, k):
    adj = data.adj
    adj_dense  = np.array(adj, dtype=np.float64)
    attribute_dense  = np.array(data.x, dtype=np.float64)
    ori_num_edge = np.sum(adj_dense)
    num_node = adj_dense.shape[0]
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    anomaly_idx = all_idx[:m]
    attribute_anomaly_idx = anomaly_idx[:]

    label = np.zeros((num_node, 1), dtype=np.uint8)
    label[anomaly_idx, 0] = 1

    attr_anomaly_label = np.zeros((num_node, 1), dtype=np.uint8)
    attr_anomaly_label[attribute_anomaly_idx, 0] = 1

    # Disturb structure
    adj_dense = to_dense_adj(data.edge_index)[0].numpy()
    # print('Constructing structured anomaly nodes...')
    # for n_ in range(n):
        # current_nodes = structure_anomaly_idx[n_*m:(n_+1)*m]
        # for i in current_nodes:
            # for j in current_nodes:
                # adj_dense[i, j] = 1.0
        # adj_dense[current_nodes,current_nodes] = 0.
        # np.fill_diagonal(adj_dense[current_nodes][:, current_nodes], 0.0)

    # num_add_edge = np.sum(adj_dense) - np.sum(to_dense_adj(data.edge_index)[0].numpy())
    # num_add_edge = np.sum(adj_dense) - ori_num_edge
    # print(f'Done. {len(structure_anomaly_idx)} structured nodes are constructed. ({num_add_edge:.0f} edges are added) \n')

    # Disturb attribute
    # attribute_dense = data.x.numpy()
    print('Constructing attributed anomaly nodes...')
    for i_ in attribute_anomaly_idx:
        picked_list = random.sample(all_idx, k)
        max_dist = 0
        for j_ in picked_list:
            cur_dist = euclidean(attribute_dense[i_], attribute_dense[j_])
            if cur_dist > max_dist:
                max_dist = cur_dist
                max_idx = j_

        attribute_dense[i_] = attribute_dense[max_idx]
    print(f'Done. {len(attribute_anomaly_idx)} attributed nodes are constructed. \n')

    data.x = torch.tensor(attribute_dense, dtype=torch.float)
    edge_index, _ = dense_to_sparse(torch.tensor(adj_dense, dtype=torch.float))
    data.edge_index = edge_index
    data.label = label

    return data
def split_subgraphs(n_clients, data, dataset):
    G = torch_geometric.utils.to_networkx(data)
    n_cuts, membership = metis.part_graph(G, n_clients)
    assert len(list(set(membership))) == n_clients
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')

    adj = to_dense_adj(data.edge_index)[0]
    for client_id in range(n_clients):
        client_indices = np.where(np.array(membership) == client_id)[0]
        client_indices = list(client_indices)
        client_num_nodes = len(client_indices)
        client_dict = {i: client_indices[i] for i in range(len(client_indices))}
        client_edge_index = []
        client_adj = adj[client_indices][:, client_indices]
        client_edge_index, _ = dense_to_sparse(client_adj)
        client_edge_index = client_edge_index.T.tolist()
        client_num_edges = len(client_edge_index)

        client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
        client_x = data.x[client_indices]
        client_y = data.y[client_indices]
        client_label = np.zeros((client_num_nodes, 1), dtype=np.uint8)
        

        client_data = Data(
            x=client_x,
            y=client_y,
            edge_index=client_edge_index.t().contiguous(),
            label=client_label,
            node_indices=client_indices,
            client_dict=client_dict,
            adj = client_adj
        )
       
        if n_clients == 5:
           
            m, k = 30, 50
        elif n_clients == 10:
          
            m, k = 15, 50
        elif n_clients == 15:
           
            m, k = 10, 50
       
        client_data = add_anomaly_nodes(client_data, m, k)
        count_ones = np.sum(client_data.label == 1)
       
        torch_save(data_path, f'{dataset}_disjoint5/{n_clients}/partition_{client_id}.pt', {
            'client_data': client_data,
            'client_id': client_id
        })
        print(f'client_id: {client_id}, iid, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')
for n_clients in clients:
    generate_data(dataset='cora', n_clients=n_clients)
    