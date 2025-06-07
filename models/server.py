import time
import numpy as np

from scipy.spatial.distance import cosine

from misc.utils import *
from models.nets import *
from modules.federated import ServerModule

from models.nets import *
from utils import preprocess_features, adj_to_dgl_graph, normalize_adj
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from data.generators.utils import get_mat_data
import scipy.sparse as sp

from torch.optim import Adam



class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)

        self.model = Model(self.args.n_feat, self.args.n_dims, 'prelu', self.args.negsamp_ratio,
                           self.args.readout).cuda(self.gpu_id)
        
        self.sd['proxy'] = self.get_proxy_data(args.n_feat)

        self.update_lists = []
        self.sim_matrices = []
        self.max_auc = 0


    def get_proxy_data(self, n_feat):
        
        import networkx as nx

       
        num_graphs, num_nodes = self.args.n_proxy, 100
    
        data = from_networkx(
            nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed))
       
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data

    def on_round_begin(self, curr_rnd):
        
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.logger.print(f'rnd: {self.curr_rnd + 1}')
        self.sd['global'] = self.get_weights()  
       
        if curr_rnd==0:
           
            file_path = f'./datasets/{self.args.dataset}.mat'
            dataset = get_mat_data(file_path)
            features, edge_index = dataset.x, dataset.edge_index
            self.ser_adj = to_dense_adj(edge_index)
            self.server_num_nodes = features.shape[0]
            num_fea = features.shape[1]
            self.num_fea = num_fea
            global_random_fea =  torch.zeros(self.server_num_nodes,self.args.n_dims) 
            from utils import normalize_adj
           
            self.sd['global_fea'] = global_random_fea
            self.sd['global_adj'] = self.ser_adj
           
        
        else:
            
            self.sd['global_fea'] = self.apply_gdc(self.sd['global_fea'], self.sd['global_adj'].squeeze(0))
        log_file_path = os.path.join(self.args.log_path, f'server.txt')
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as file:
                data = json.load(file)
                self.log = data.get('log', {'rnd_auc': [], 'rnd_threshold': []})
        else:
            self.log = {
                'rnd_server_auc': [],
            }
        

    def on_round_complete(self, updated):
        
        avg_auc = 0
        for c_id in updated:
            avg_auc += self.sd[f'{c_id}_auc']

        
        avg_auc = avg_auc / len(updated)
        self.max_auc = avg_auc if avg_auc > self.max_auc else self.max_auc
        
        self.args.max_auc = self.max_auc
        
        self.logger.print('Avg_AUC:{:.4f}, Max_AUC: {:.4f}'.format(avg_auc, self.max_auc))
        self.log['rnd_server_auc'].append(avg_auc)
        self.update(updated)
        self.save_state() 
        self.save_log()

    def update(self, updated):
        

        st = time.time()

        local_weights = []
        local_functional_embeddings = []
        local_train_sizes = []

        for c_id in updated:
            local_weights.append(self.sd[c_id]['model'].copy())
            local_functional_embeddings.append(self.sd[c_id]['functional_embedding'][0])
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]

       

        n_connected = round(self.args.n_clients * self.args.frac)
        assert n_connected == len(local_functional_embeddings)
       
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        
        for i in range(n_connected):
            for j in range(n_connected):
                sim_matrix[i, j] = 1 - cosine(local_functional_embeddings[i], local_functional_embeddings[j])

        if self.args.agg_norm == 'exp':
            sim_matrix = np.exp(self.args.norm_scale * sim_matrix)

        
        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]

        st = time.time()

        
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio))

        self.logger.print(f'global model has been updated ({time.time() - st:.2f}s)')

        st = time.time()

       
        for i, c_id in enumerate(updated):
            aggr_local_model_weights = self.aggregate(local_weights, sim_matrix[i, :])
            if f'personalized_{c_id}' in self.sd: del self.sd[f'personalized_{c_id}']
            self.sd[f'personalized_{c_id}'] = {'model': aggr_local_model_weights}

        self.update_lists.append(updated)
        self.sim_matrices.append(sim_matrix)
       

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model),
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'sim_matrices': self.sim_matrices,
            'update_lists': self.update_lists
        })


    def generate_data(self):
        if self.curr_rnd == 0:
            for c_id in range(self.args.n_clients):
                self.loader.switch(c_id)
                dataset = self.loader.partition
                c_fea = dataset.feat
                adj = dataset.adj
                adj = adj.todense()
                adj = torch.tensor(adj, dtype=torch.float32) 
                nb_nodes = c_fea.shape[0]  
                nb_fea = c_fea.shape[1]
                global_random_fea = torch.zeros(nb_nodes, self.args.n_dims)
                self.sd[f'global{c_id}_fea'] = global_random_fea
                self.sd[f'global{c_id}_adj'] = adj
        elif self.curr_rnd > 0:  
           
            for c_id in range(self.args.n_clients):
                self.sd[f'global{c_id}_fea'] = self.apply_gdc(self.sd[f'global{c_id}_fea'], self.sd[f'global{c_id}_adj'])


       

    def apply_gdc(self, features, adj):
        from torch_geometric.data import Data
        from torch_geometric.transforms import GDC 
       
        data = Data(x=features, edge_index=dense_to_sparse(adj)[0])
      
        gdc = GDC(diffusion_kwargs=dict(method='ppr', alpha=self.args.ppr_alpha),
                  sparsification_kwargs=dict(method='topk', k=128, dim=0),
                  exact=True)
        data = gdc(data)
        return data.x

