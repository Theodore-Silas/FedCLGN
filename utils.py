"""
FileName: 
Author: 
Version:
Description: 
"""
import numpy as np
import networkx as nx # 2.3 2.6.3
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
import os

from scipy.sparse import csc_matrix
from scipy.spatial.distance import euclidean
from torch.nn.functional import cosine_similarity
def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """
    Generate subgraph with RWR algorithm(重启随机游走)
    subgraph_size:子图大小
    """
    all_idx = list(range(dgl_graph.number_of_nodes()))  # 获取节点索引列表
    reduced_size = subgraph_size - 1  # 减1，是为了后面增加初始节点
    # 生成随机游走路径
    # 当客户端数量增多，可能出现no successors from vertex 91
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1,
                                                           max_nodes_per_seed=subgraph_size * 3)
    subv = []
    for i, trace in enumerate(traces):  # 遍历每个游走路径
        # 将节点的邻居节点存入子图节点集合当中
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:  # 如果节点数量不满足大小，重新RWR
            # 将每条路径的
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                      max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]  # 将每个节点的游走序列截断到reduced_size的大小
        subv[i].append(i)  # 确保每个子图都包含起始节点,subv[i]大小为subgrpah_size
    # transformed_subv = [[nodes[idx] for idx in sublist] for sublist in subv]
    return subv  # 返回每个子图的节点索引列表


def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)
    return torch.load(fpath, map_location=torch.device('cpu'))

def adj_to_dgl_graph(adj):
    """
    Convert adj to dgl format.
    """
    nx_graph = nx.from_numpy_array(adj)
    # nx_graph =nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

def adj_to_dgl_graph1(adj, node_indices):
    nx_graph = nx.from_numpy_array(adj)

    # Relabel the nodes in the graph with the provided node indices
    mapping = {i: node_indices[i] for i in range(len(node_indices))}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)

    # Convert the NetworkX graph to a DGL graph
    # edges = list(nx_graph.edges())
    # src, dst = zip(*edges)
    dgl_graph = dgl.DGLGraph(nx_graph)
    # dgl_graph = dgl.DGLGraph((src,dst))
    return nx_graph

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """
    Convert sparse matrix to tuple representation
    insert_batch:决定是否插入一个批次的维度
    """

    def to_tuple(mx):  # 将单个系数矩阵转换为元组表示
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()  # 转为coo格式
        # 例如稀疏矩阵data = [0,0,1][2,0,0],[0,3,0]
        # coords:稀疏矩阵中非零元素的坐标
        # values:对应坐标位置的值
        # shape:稀疏矩阵的形状
        if insert_batch:  # if you want to insert a batch dimension
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):  # 如果是存储稀疏矩阵的列表
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx  # 可以获取.data,.row,.col,.shape
def preprocess_features(features):
    """
    Row-normalize feature matrix and convert
    归一化特征矩阵
    """
    rowsum = np.array(features.sum(1))  # 计算矩阵每一行的和，形成一个列向量
    r_inv = np.power(rowsum, -1).flatten()  # rowsum取倒数，展平为一维数组
    r_inv[np.isinf(r_inv)] = 0.  # 无穷大值置为0
    r_mat_inv = sp.diags(r_inv)  # 创建一个对角阵
    features = r_mat_inv.dot(features)  # 使用对角阵乘以原矩阵，归一化
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    归一化邻接矩阵
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # 最终实现对称归一化 A= D^-1/2 A D^-1/2

def cosine_simi(tensor1, tensor2):
    '''

    :param tensor1:
    :param tensor2:
    :return: 相似度的标量
    '''
    return cosine_similarity(tensor1, tensor2, dim=2)


####自定义实现重启随机游走
def custom_random_walk_with_restart(dgl_graph, start_nodes, restart_prob, max_nodes_per_seed, anomalous_indices):
    traces = []
    for start_node in start_nodes:
        trace = [start_node]
        current_node = start_node
        for _ in range(max_nodes_per_seed):
            if torch.rand(1).item() < restart_prob:
                current_node = start_node
            # 获取邻居
            neighbors = dgl_graph.successors(current_node).tolist()
            # 排除异常节点索引
            neighbors = [node for node in neighbors if node not in anomalous_indices]
            if not neighbors:
                current_node = start_node
            else:
                current_node = neighbors[torch.randint(0, len(neighbors), (1,)).item()]
                trace.append(current_node)
        traces.append(trace)
    return traces

def generate_rwr_subgraph1(dgl_graph, subgraph_size, anomalous_indices):
    all_idx = list(range(dgl_graph.number_of_nodes()))  # Get list of all node indices
    # reduced_size = subgraph_size - 1  # Reduced size to add initial node later
    reduced_size = subgraph_size  # Reduced size to add initial node later
    traces = custom_random_walk_with_restart(dgl_graph, all_idx, restart_prob=1,
                                             max_nodes_per_seed=subgraph_size * 3, anomalous_indices=anomalous_indices)

    subv = []
    for i, trace in enumerate(traces):  # Iterate over each random walk path
        unique_nodes = torch.unique(torch.tensor(trace), sorted=False).tolist()
        retry_time = 0
        while len(unique_nodes) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                      max_nodes_per_seed=subgraph_size * 5)
        # Retry if subgraph size is not met
        #     cur_trace = custom_random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
        #                                                 max_nodes_per_seed=subgraph_size * 5,
        #                                                 anomalous_indices=anomalous_indices)
            unique_nodes = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(unique_nodes) <= 2) and (retry_time > 10):  # Avoid infinite loop
                unique_nodes = unique_nodes * reduced_size
        unique_nodes = unique_nodes[:reduced_size]  # Trim to reduced_size
        # unique_nodes.append(i)  # Ensure the initial node is included

        unique_nodes = unique_nodes[:reduced_size]  # Trim to reduced_size
        first_element = unique_nodes.pop(0)
        unique_nodes.append(first_element)
        subv.append(unique_nodes)

    return subv  # Return the list of subgraph node indices