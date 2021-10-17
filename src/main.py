import argparse
import numpy as np
import time
from data_loader import load_data
from train import train, test
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# np.random.seed(777)


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='one_week', help='which dataset to use')
parser.add_argument('--title_len', type=int, default=10, help='the max length of title')
parser.add_argument('--session_len', type=int, default=10, help='the max length of session')
parser.add_argument('--aggregator', type=str, default='neighbor', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=5, help='the number of epochs')
parser.add_argument('--user_neighbor', type=int, default=30, help='the number of neighbors to be sampled')
parser.add_argument('--news_neighbor', type=int, default=10, help='the number of neighbors to be sampled')
parser.add_argument('--entity_neighbor', type=int, default=1, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=128, help='dimension of user and entity embeddings')
parser.add_argument('--cnn_out_size', type=int, default=128, help='dimension of cnn output')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=5e-3, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--topic', type=int, default=1, help='whether use topic')
parser.add_argument('--n_topics', type=int, default=20, help='numble of topics')
parser.add_argument('--save_path', type=str, default="../data/10week/hop2", help='model save path')
parser.add_argument('--test', type=int, default=0, help='test')
parser.add_argument('--use_group', type=int, default=1, help='whether use group')
parser.add_argument('--n_filters', type=int, default=64, help='number of filters for each size in KCNN')
parser.add_argument('--filter_sizes', type=int, default=[2, 3], nargs='+',
                    help='list of filter sizes, e.g., --filter_sizes 2 3')

show_loss = True
show_time = False

# t = time()

args = parser.parse_args()
data = load_data(args)
if args.test != 1:
    train(args, data, show_loss)
else:
    test(args, data)

if show_time:
    print('time used: %d s' % (time() - t))