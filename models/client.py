import json
import os.path
import time
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from misc.utils import *
from models.nets import *
from modules.federated import ClientModule
from utils import *
import scipy.sparse as sp



class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.model = Model(self.args.n_feat, self.args.n_dims, 'prelu', self.args.negsamp_ratio,
                           self.args.readout).cuda(g_id)
        self.parameters = list(self.model.parameters())
        self.model_path = self.args.log_path


    def init_state(self):
        
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        
        log_file_path = os.path.join(self.args.log_path, f'client_{self.client_id}.txt')
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as file:
                data = json.load(file)
                self.log = data.get('log', {'rnd_auc': [], 'rnd_threshold': []})
        else:
            self.log = {
            'rnd_auc': [],
            'rnd_threshold': []
            }

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
        })
        # torch.save(self.model.state_dict(), f'{self.args.checkpt_path}/{self.client_id}_best_model.pkl')



    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']
    
    def on_receive_message(self, curr_rnd):
        
        self.curr_rnd = curr_rnd
        
        self.update(self.sd[f'personalized_{self.client_id}' \
            if (f'personalized_{self.client_id}' in self.sd) else 'global'])
        
        self.global_w = convert_np_to_tensor(self.sd['global']['model'], self.gpu_id)

    def update(self, update):
       
        self.prev_w = convert_np_to_tensor(update['model'], self.gpu_id)
        
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True)

    def on_round_begin(self):
        self.train()
        self.transfer_to_server()  

    def get_sparsity(self):
        
        n_active, n_total = 0, 1
        for mask in self.masks:
            pruned = torch.abs(mask) < self.args.l1
            mask = torch.ones(mask.shape).cuda(self.gpu_id).masked_fill(pruned, 0)
            n_active += torch.sum(mask)
            _n_total = 1
            for s in mask.shape:
                _n_total *= s
            n_total += _n_total
        return ((n_total-n_active)/n_total).item()

    def train(self):
        st = time.time()
        
        if self.curr_rnd == 0:
            self.alpha = 1 
        else:
            self.alpha = 0.15
        self.cola_train(self.alpha)
    def cola_train(self, alpha):
        dataset = self.loader.partition
        adj, features, labels, ano_label, client_dict  = dataset.adj, dataset.feat, dataset.labels, dataset.ano_labels, dataset.client_dict
        
        node_idx_key = dataset.client_dict

       
        num_ones = np.sum(ano_label == 1)
        
        total_length = len(ano_label)
        
        self.proportion_ones = num_ones / total_length if total_length > 0 else 0
        
        if self.curr_rnd > 0:
            neg_fea = self.sd[f'global_fea'].unsqueeze(0)
           
            neg_fea = neg_fea.cuda()


        features, _ = preprocess_features(features)  
        dgl_graph = adj_to_dgl_graph(adj)
        nb_nodes = features.shape[0]  
        ft_size = features.shape[1] 
        adj = normalize_adj(adj)
        adj = (adj + sp.eye(adj.shape[0])).todense()  
        
        features = torch.FloatTensor(features[np.newaxis])
        adj = torch.FloatTensor(adj[np.newaxis])
        labels = torch.FloatTensor(labels[np.newaxis])


        if torch.cuda.is_available():
           
            self.model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
           

        batch_size = self.args.batch_size
        subgraph_size = self.args.subgraph_size
        
        batch_num = (nb_nodes + batch_size - 1) // batch_size  
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0  
        best = 1e9 
        best_t = 0  
        if torch.cuda.is_available():
            
            b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([self.args.negsamp_ratio]).cuda())
        else:
            b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([self.args.negsamp_ratio]))
        added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
        added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
        added_adj_zero_col[:, -1, :] = 1.
        added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))
        if torch.cuda.is_available():
            added_adj_zero_row = added_adj_zero_row.cuda()
            added_adj_zero_col = added_adj_zero_col.cuda()
            added_feat_zero_row = added_feat_zero_row.cuda()
        num_epoch = self.args.num_epoch
        epoch_bfs = []
        
        for epoch in range(num_epoch):
            bfs = []
            h_1_batches = []
           
            loss_full_batch = torch.zeros(nb_nodes, 1) 
            if torch.cuda.is_available():
                loss_full_batch = loss_full_batch.cuda()

            self.model.train()  
           
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            
            total_loss = 0
           
            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
           
            for batch_idx in range(batch_num):
                batch_node_embeddings = {} 
                
                self.optimizer.zero_grad()
                is_final_batch = (batch_idx == (batch_num - 1))  
               
                if not is_final_batch:  
                    idx = all_idx[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    

                else:
                    idx = all_idx[batch_idx * batch_size:]
                
                cur_batch_size = len(idx)
                alpha2 = torch.zeros(cur_batch_size, 64)  
                
                lbl = torch.unsqueeze(
                    torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * self.args.negsamp_ratio))),
                    1)

                ba = []  
                bf = []  

                neg_ba = []
                neg_bf = []
               
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()
                
                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)
                    if self.curr_rnd == 0:
                        cur_neg_feat = features[:, subgraphs[i], :]
                    if self.curr_rnd > 0:
                        
                        neg_idx = []
                        for kk in subgraphs[i]:
                            neg_idx.append(node_idx_key[kk])
                        cur_neg_feat = neg_fea[:, neg_idx, :]
                        
                    cur_neg_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    neg_ba.append(cur_neg_adj)
                    neg_bf.append(cur_neg_feat)
                
                ba = torch.cat(ba)  
                ba = torch.cat((ba, added_adj_zero_row), dim=1)  
                ba = torch.cat((ba, added_adj_zero_col), dim=2)  
                bf = torch.cat(bf)  
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)  
                bfs.append(bf)
                neg_ba = torch.cat(neg_ba)
               
                neg_ba = torch.cat((neg_ba, added_adj_zero_row), dim=1)
                neg_ba = torch.cat((neg_ba, added_adj_zero_col), dim=2)
                neg_bf = torch.cat(neg_bf)
                
                if self.curr_rnd == 0:
                    neg_bf = torch.cat((neg_bf[:, :-1, :], added_feat_zero_row, neg_bf[:, -1:, :]), dim=1)
               
                if self.curr_rnd == 0:
                    logits = self.model(bf, ba, neg_bf, neg_ba, alpha, firstrnd=True)
                else:
                    logits = self.model(bf, ba, neg_bf, neg_ba, alpha)
                
                loss_all = b_xent(logits, lbl)  
              
                loss = torch.mean(loss_all) 
                
                loss.backward() 
                self.optimizer.step()
                loss = loss.detach().cpu().numpy()  
                loss_full_batch[idx] = loss_all[: cur_batch_size].detach()  
                if not is_final_batch:
                    total_loss += loss 
                

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
            epoch_bfs.append(bfs)
            if mean_loss < best:
                best = mean_loss  
                best_t = epoch
                cnt_wait = 0
                self.save_state()
                # torch.save(self.model.state_dict(), f'{self.args.checkpt_path}/{self.client_id}_best_model.pkl')
            else:
                cnt_wait += 1  


        multi_round_ano_score = np.zeros((self.args.auc_test_rounds, nb_nodes))
        
            
        for round in range(self.args.auc_test_rounds):

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            
            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
           
            for batch_idx in range(batch_num):

                self.optimizer.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)
                alpha2 = torch.zeros(cur_batch_size, 64)  # 权重矩阵 (300, 64)
               
                ba = []
                bf = []
                neg_ba = []
                neg_bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)
                    if self.curr_rnd ==0:
                        cur_neg_feat = features[:, subgraphs[i], :]
                    if self.curr_rnd > 0:
                       
                        neg_idx = []
                        for kk in subgraphs[i]:
                            neg_idx.append(node_idx_key[kk])
                        cur_neg_feat = neg_fea[:, neg_idx, :]

                      
                    cur_neg_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    neg_ba.append(cur_neg_adj)
                    neg_bf.append(cur_neg_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
                neg_ba = torch.cat(neg_ba)
                neg_ba = torch.cat((neg_ba, added_adj_zero_row), dim=1)
                neg_ba = torch.cat((neg_ba, added_adj_zero_col), dim=2)
                neg_bf = torch.cat(neg_bf)
                if  self.curr_rnd == 0:
                    neg_bf = torch.cat((neg_bf[:, :-1, :], added_feat_zero_row, neg_bf[:, -1:, :]), dim=1)

                with torch.no_grad():
                    if self.curr_rnd == 0:
                        logits = self.model(bf, ba, neg_bf, neg_ba, alpha, firstrnd=True)
                    else:
                        logits = self.model(bf, ba, neg_bf, neg_ba,alpha)  
                   
                    logits = torch.squeeze(logits)
                    logits = torch.sigmoid(logits)

                ano_score = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()

                multi_round_ano_score[round, idx] = ano_score
               

            
        ano_score_final = np.mean(multi_round_ano_score, axis=0)
        auc = roc_auc_score(ano_label, ano_score_final)
       
        ano_mask = ano_score_final >= self.args.ano_score_threshold 
        ano_mask = list(np.nonzero(ano_mask)[0])
        self.model.eval()
        ano_out = self.model(features,adj,features,adj,self.alpha, is_proxy=True)
        for ano_idx in ano_mask:
            
            ano_raw_idx = dataset.client_dict[ano_idx]
            self.sd['global_fea'][ano_raw_idx, :] = ano_out[:, ano_idx, :].squeeze(0).cpu().clone()  # upload
           
       
        self.sd[f'{self.client_id}_auc'] = auc
        
        self.log['rnd_auc'].append(auc)
       
        self.save_log()
        

    @torch.no_grad()
    def get_functional_embedding(self):
        self.model.eval()
        with torch.no_grad():
            proxy_in = self.sd['proxy']
            proxy_in = proxy_in.cuda(self.gpu_id)
            from torch_geometric.utils import to_dense_adj
            x = proxy_in.x.unsqueeze(0)
            adj = to_dense_adj(proxy_in.edge_index)
            if torch.cuda.is_available():
                x = x.cuda()
                adj = adj.cuda()
                
            proxy_out = self.model(x, adj, x, adj, self.alpha, is_proxy=True)

            proxy_out = proxy_out.mean(dim=0)
            proxy_out = proxy_out.clone().detach().cpu().numpy()
        return proxy_out

    def transfer_to_server(self):
        

        self.sd[self.client_id] = {
            'model': get_state_dict(self.model),
            'train_size': self.loader.partition.len,
            'ano_pro': self.proportion_ones,
            'functional_embedding': self.get_functional_embedding()

        }


   



    
    
