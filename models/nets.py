import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0) 
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class AvgReadout(nn.Module):
   
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
   
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values #

class MinReadout(nn.Module):
   
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        
        query = query.permute(0,2,1) 
        sim = torch.matmul(seq, query) 
        sim = F.softmax(sim, dim=1) 
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim) 
        out = torch.sum(out, 1) 

        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1) 
        self.negsamp_round = negsamp_round
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


    def forward(self, c, h_pl, neg_c, neg_h_mv, alpha):
        scs = []  
       
        c_mi = c
        neg_c_mi = neg_c
        c_fused = alpha * c_mi  + (1-alpha)*neg_c_mi

       
        a = self.f_k(h_pl, c_fused)

        scs.append(a)  

        cos_sim = torch.cosine_similarity(c_mi, neg_c_mi, dim=1)
       
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            neg_c_mi = torch.cat((neg_c_mi[-2:-1, :], neg_c_mi[:-1, :]), 0)
          
            fused = alpha * c_mi + (1 - alpha) * neg_c_mi
            
            d = self.f_k(h_pl, fused) 
          
            scs.append(d)
         

        logits = torch.cat(tuple(scs))

        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn = GCN(n_in, n_h, activation)
     
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()
      
        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj, glob_neg_seq, glob_neg_adj, alpha, sparse=False, is_proxy=False, firstrnd=False):

        h_1 = self.gcn(seq1, adj, sparse)  
        if is_proxy:
            return h_1
        
        if firstrnd:
            h_2 = self.gcn(glob_neg_seq, glob_neg_adj, sparse) 
        else:
            h_2 = glob_neg_seq
        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:, : -1, :]) 
            h_mv = h_1[:, -1, :]  
            neg_h_mv = h_2[:, -1, :]  
            neg_c = self.read(h_2[:, :-1, :]) 

        else:
            h_mv = h_1[:, -1, :] 
            neg_h_mv = h_2[:, -1, :] 
            c = self.read(h_1[:, : -1, :], h_1[:, -2: -1, :])
            neg_c = self.read(h_2[:, :-1, :], h_2[:, -2: -1, :]) 
        ret = self.disc(c, h_mv, neg_c, neg_h_mv, alpha) 
      
        return ret

class MaskedGCN(nn.Module):
        def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
           
            super().__init__()
            self.n_feat = n_feat
            self.n_dims = n_dims
            self.n_clss = n_clss
            self.args = args

            from models.layers import MaskedGCNConv, MaskedLinear
            self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1, args=args)
            self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1, args=args)
            self.clsif = MaskedLinear(self.n_dims, self.n_clss, l1=l1, args=args)

        def forward(self, data, is_proxy=False):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            if is_proxy == True: return x  
         
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.clsif(x)
            return x

class GraphDiffusionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K=10, **kwargs):
        super(GraphDiffusionConv, self).__init__(aggr='add', **kwargs)
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.K = K  

    def forward(self, x, edge_index, sparse=False):
      
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

        for _ in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
        return self.lin(x)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out