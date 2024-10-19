import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to use (music, )')
parser.add_argument('--n_epoch', type=int, default=70, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--n_layer', type=int, default=2, help='depth of layer')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-8, help='weight of the l2 regularization term')
parser.add_argument('--dim', type=int, default=128, help='dimension of entity and relation embeddings')
parser.add_argument('--kg_triple_set_size', type=int, default=128, help='the number of triples in triple set of kg')
parser.add_argument('--kg_potential_triple_set_sampling_size', type=int, default=128, help='the number of triples in triple set of kg potential set')
parser.add_argument('--ddi_origin_triple_set_size', type=int, default=128, help=' the number of triples in triple set of ddi origin ' )
parser.add_argument('--ddi_potential_triple_set_sampling_size', type=int, default=128, help='the number of triples in triple set of ddi' )
parser.add_argument('--agg', type=str, default='concat', help='the type of aggregation function (sum, pool, concat)')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')

parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

args = parser.parse_args()


def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)                  
    torch.manual_seed(torch_seed)       
    torch.cuda.manual_seed(torch_seed)      
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enable = True

if not args.random_flag:
    set_random_seed(304, 2018)

data_info = load_data(args)
train(args, data_info)
