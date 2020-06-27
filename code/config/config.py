import math
import argparse
import torch

from util.train_util import *
from util.parameter_util import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-e', '--epochs', type=int, dest='train_epochs', help='total train epochs', required=False, default=21)
parser.add_argument('-b', '--batchsize', type=int, dest='batchsize', help='batch size', required=False, default=500)
parser.add_argument('-m', '--margin', type=float, dest='margin', help='margin', required=False, default=10.0)
parser.add_argument('-l', '--learning_rate', type=float, dest="learning_rate", help="learning rate", required=False, default=0.005)
parser.add_argument('-d', '--dimension', type=int, dest="dimension", help="dimension", required=False, default=100)
parser.add_argument('-n', '--norm', type=int, dest="norm", help="normalization", required=False, default=2)
parser.add_argument('-o', '--optim', type=str, dest="optim", help="optimizer", required=False, default="SGD")
parser.add_argument('-p', '--path', type=str, dest="dataset", help="dataset path", required=False, default="YAGO-3SP/snapshot1")
parser.add_argument('-t', '--test_mode', type=bool, dest="test_mode", help="if test mode on", required=False, default=False)
args = parser.parse_args()


dataset_v1 = args.dataset

entity_total = get_total(file_name='./data/' + dataset_v1 + '/entity2id.txt')
relation_total = get_total(file_name='./data/' + dataset_v1 + '/relation2id.txt')
entity_set = set(range(entity_total))
relation_set = set(range(relation_total))

train_list = read_file(file_name='./data/' + dataset_v1 + '/train2id.txt')
test_list = read_file(file_name='./data/' + dataset_v1 + '/test2id.txt')

print('entity_total: ' + str(entity_total))
print('relation_total: ' + str(relation_total))
print('train_total: ' + str(len(train_list)))
print('test_total: ' + str(len(test_list)))

max_context = 30
train_times = args.train_epochs
learning_rate = args.learning_rate
batch_size = args.batchsize
nbatchs = math.ceil(len(train_list) / batch_size)
margin = args.margin
dim = args.dimension
norm = args.norm
optimizer = args.optim
test_mode = args.test_mode
bern = True
res_dir = './data/' + dataset_v1 + '/parameters/'

print('train_times: ' + str(train_times))
print('learning_rate: ' + str(learning_rate))
print('batch_size: ' + str(batch_size))
print('nbatchs: ' + str(nbatchs))
print('dim: ' + str(dim))
print('margin: ' + str(margin))
print('bern: ' + str(bern))
print('result directory: ' + str(res_dir))

if not test_mode:
    print("Constructing adj table...")
    entity_adj_table, relation_adj_table, max_context_num, entity_A, relation_A = construct_adj_table(train_list, entity_total,
                                                                                                      relation_total, max_context)
    print("Constructing adj table completed.")


def prepare_data():
    phs = np.zeros(len(train_list), dtype=int)
    prs = np.zeros(len(train_list), dtype=int)
    pts = np.zeros(len(train_list), dtype=int)
    nhs = np.zeros((train_times, len(train_list)), dtype=int)
    nrs = np.zeros((train_times, len(train_list)), dtype=int)
    nts = np.zeros((train_times, len(train_list)), dtype=int)

    train_set = set(train_list)
    tph, hpt = bern_sampling_prepare(train_list)
    for i, golden_triple in enumerate(train_list):
        # print(i, end="\r")
        phs[i], prs[i], pts[i] = golden_triple

        for j in range(train_times):
            negative_triples = one_negative_sampling(golden_triple, train_set, entity_total, True, tph, hpt)
            nhs[j][i], nrs[j][i], nts[j][i] = negative_triples

    return torch.IntTensor(phs).cuda(), torch.IntTensor(prs).cuda(), torch.IntTensor(pts).cuda(), torch.IntTensor(
        nhs).cuda(), torch.IntTensor(nrs).cuda(), torch.IntTensor(nts).cuda()
