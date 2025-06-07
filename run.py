import os
from datetime import datetime

import torch.backends.cudnn
from models.server import Server
from models.client import Client
from misc.utils import *
from modules.multiprocs import ParentProcess
import dgl
import warnings

warnings.filterwarnings("ignore")


class Args:
    def __init__(self):
       
        self.gpu = '3'
        self.n_workers = 1  
        self.dataset = 'Cora' 
        self.mode = 'disjoint5'  
        self.frac = 1 
        self.n_rnds = 50  
        self.n_eps = 1  
        self.n_clients = 5  
        self.seed = 1  
        self.clsf_mask_one = False  
        self.laye_mask_one = False  
        self.norm_scale = None  
        self.base_path = os.getcwd() 
        self.debug = False

        self.n_feat = None
        self.n_clss = None
        self.n_dims = 64
        self.base_lr = None  
        self.min_lr = None  
        self.momentum_opt = None  
        self.weight_decay = None  
        self.warmup_epochs = None  
        self.base_momentum = None  
        self.final_momentum = None  
        self.loc_l2 = 1e-3
        self.l1 = 1e-3
       
        self.readout = 'avg'  
        self.negsamp_ratio = 1
        self.subgraph_size = 4
        self.batch_size = 60  
        self.num_epoch = 15  
        self.auc_test_rounds = 15 

        self.max_auc = 0
        self.ppr_alpha = 0.5  
        self.ano_score_threshold = -0.68 
        
        self.c_cost = []  #cost

def main(args):
    args = set_config(args)
    
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHSAHSEED'] = str(args.seed)
    os.environ['OMP_NUM_THREADS'] = '1' 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.enabled = True
    
    pp = ParentProcess(args, Server, Client)  
    pp.start()

    return args.max_auc 

def set_config(args):
   
    args.base_lr = 1e-3  
    args.min_lr = 1e-3
    args.momentum_opt = 0.9
    args.weight_decay = 1e-5  
    args.warmup_epochs = 10
    args.base_momentum = 0.99
    args.final_momentum = 1.0

    if args.dataset == 'Cora':
        args.n_feat = 1433
        args.n_clss = 7
    if args.dataset == 'Citeseer':
        args.n_feat = 3703
    if args.dataset == 'BlogCatalog':
        args.n_feat = 8189
        args.base_lr = 3e-3
        #args.num_epoch = 50
        #args.auc_test_rounds = 50
    if args.dataset =='Flickr':
        args.n_feat = 12047
       

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    trial = f'{args.dataset}/Clients_{args.n_clients}/{now}'

  
    args.data_path = os.path.join(args.base_path, 'datasets')
    args.checkpt_path = os.path.join(args.base_path, 'checkpoints', trial)
    args.log_path = os.path.join(args.base_path, 'logs', trial)

    if args.debug:
        args.checkpt_path = os.path.join(args.base_path, 'debug', 'checkpoints', trial)
        args.log_path = os.path.join(args.base_path, 'debug', 'logs', trial)
 
    dataset_path = args.data_path
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
    args.data_path = dataset_path

    return args

    

if __name__ == '__main__':
    
    torch.multiprocessing.set_start_method('spawn')
    args = Args()
    args.clsf_mask_one = True
    args.laye_mask_one = True
    args.norm_scale = 3
    args.n_proxy = 5
    args.agg_norm = 'exp'
    args.dataset = 'Cora'
    args.n_clients = 5
    args.gpu = '3'
    
    auc = main(args)
