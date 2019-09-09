import torch
import torch.nn as nn
import pickle
import os
import random
import math
import multiprocessing
import numpy as np
import json
import operator
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-b', '--batchsize', type=int, dest='batchsize', help='batch size', required=False, default=200)
parser.add_argument('-m', '--margin', type=float, dest='margin', help='margin', required=False, default=10.0)
parser.add_argument('-l', '--learning_rate', type=float, dest="learning_rate", help="learning rate", required=False, default=0.005)
parser.add_argument('-d', '--dimension', type=int, dest="dimension", help="dimension", required=False, default=100)
parser.add_argument('-n', '--norm', type=int, dest="norm", help="normalization", required=False, default=2)
parser.add_argument('-e', '--extra', type=str, dest="extra", help="extra information", required=False, default="")
parser.add_argument('-t', '--test', type=int, dest="test_epoch", help="test epoch", required=False, default=0)
args = parser.parse_args()


def get_total(file_name):
    with open(file_name) as f:
        return int(f.readline())


def get_basic_info(train_set):
    entity_set = set()
    relation_set = set()
    entity_context_dict = dict()
    for (h, r, t) in train_set:
        entity_set.add(h)
        entity_set.add(t)
        relation_set.add(r)

        entity_context_dict.setdefault(h, set()).add(t)
        entity_context_dict.setdefault(t, set()).add(h)
    return entity_set, relation_set, entity_context_dict


def construct_text2id_dict(file_name):
    result_dict = dict()
    with open(file_name) as f:
        for line in f.readlines()[1:]:
            text, id = line.rstrip("\n").split("\t")[:]
            result_dict[text] = int(id)
    return result_dict


def construct_id2text_dict(file_name):
    result_dict = dict()
    with open(file_name) as f:
        for line in f.readlines()[1:]:
            text, id = line.rstrip("\n").split("\t")[:]
            result_dict[int(id)] = text
    return result_dict


def convert_id_to_text(train_list, dataset):
    text_train_list = list()
    entityid2text_dict = construct_id2text_dict('./data/' + dataset + '/entity2id.txt')
    relationid2text_dict = construct_id2text_dict('./data/' + dataset + '/relation2id.txt')
    for (h, r, t) in train_list:
        text_train_list.append((entityid2text_dict[h], relationid2text_dict[r], entityid2text_dict[t]))
    return text_train_list


def read_file(file_name):
    # for reading train, test and valid data
    data = []  # [(h, r, t)]
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            li = line.split()
            if len(li) == 3:
                data.append((int(li[0]), int(li[2]), int(li[1])))
    return data


dataset_v1 = 'DBpedia'
# dataset_v2 = 'IMDB/snapshot9'

entity_total = get_total(file_name='./data/' + dataset_v1 + '/entity2id.txt')
relation_total = get_total(file_name='./data/' + dataset_v1 + '/relation2id.txt')
entity_set = set(range(entity_total))
relation_set = set(range(relation_total))

train_list = read_file(file_name='./data/' + dataset_v1 + '/train2id.txt')
test_list = read_file(file_name='./data/' + dataset_v1 + '/test2id.txt')
valid_list = read_file(file_name='./data/' + dataset_v1 + '/valid2id.txt')

print('entity_total: ' + str(entity_total))
print('relation_total: ' + str(relation_total))
print('train_total: ' + str(len(train_list)))
print('test_total: ' + str(len(test_list)))
print('valid_total: ' + str(len(valid_list)))

train_times = 2001
validation_step = 20
max_context = 30
norm = args.norm
learning_rate = args.learning_rate
batch_size = args.batchsize
nbatchs = math.ceil(len(train_list) / batch_size)  # 单例输入，等于训练数据的数目
dim = args.dimension
margin = args.margin
extra_info = args.extra
test_epoch = args.test_epoch
bern = True
init_with_DKGE = False
init_with_transe = True
max_context_num_constraint = True
transe_model_file = 'TransE.json'
res_dir = "./res/%s_%s_%s_%s_%s_%s/" % (str(norm), str(batch_size), str(margin),
                                         str(dim), str(learning_rate), extra_info)

print('train_times: ' + str(train_times))
print('validation_step: ' + str(validation_step))
print('learning_rate: ' + str(learning_rate))
print('batch_size: ' + str(batch_size))
print('nbatchs: ' + str(nbatchs))
print('dim: ' + str(dim))
print('margin: ' + str(margin))
print('bern: ' + str(bern))
print('init_with_transe: ' + str(init_with_transe))
print('result directory: ' + str(res_dir))


# {entity: [[path]}
def get_1or2_path_from_head(head_ent, rel, entity_adj_table_with_rel):
    paths = {}  # actually second-order + first-order, {entity: [[edge]]}
    first_order_entity = set()
    first_order_relation = dict()

    if head_ent not in entity_adj_table_with_rel:
        return paths
    for tail_entity, relation in entity_adj_table_with_rel[head_ent]:
        first_order_entity.add(tail_entity)
        if relation != rel:
            if tail_entity in paths:
                paths[tail_entity].append([relation])
            else:
                paths[tail_entity] = [[relation]]

        if tail_entity in first_order_relation:
            first_order_relation[tail_entity].append(relation)
        else:
            first_order_relation[tail_entity] = [relation]

    for node in first_order_entity:
        if node not in entity_adj_table_with_rel:
            continue
        for tail_entity, relation in entity_adj_table_with_rel[node]:
            if tail_entity in paths:
                for r in first_order_relation[node]:
                    paths[tail_entity].append([r, relation])
            else:
                paths[tail_entity] = []
                for r in first_order_relation[node]:
                    paths[tail_entity].append([r, relation])

    return paths  # {entity: [[edge]]}


def find_relation_context(h, r, t, entity_adj_table_with_rel):
    #{entity: [[edge]]}
    tail_ent2paths = get_1or2_path_from_head(h, r, entity_adj_table_with_rel)
    return tail_ent2paths.get(t, [])


def construct_adj_table(train_list):
    entity_adj_table_with_rel = dict()  # {head_entity: [(tail_entity, relation)]}
    entity_adj_table = dict()  # {head_entity: [tail_entity]}
    relation_adj_table = dict()  # {relation: [[edge]]}

    for train_data in train_list:
        h, r, t = train_data
        entity_adj_table.setdefault(h, set()).add(t)
        entity_adj_table.setdefault(t, set()).add(h)
        entity_adj_table_with_rel.setdefault(h, list()).append((t, r))
        # if h not in entity_adj_table:
        #     entity_adj_table[h] = {t}
        #     entity_adj_table_with_rel[h] = [(t, r)]
        # else:
        #     entity_adj_table[h].add(t)
        #     entity_adj_table_with_rel[h].append((t, r))

    for train_data in train_list:
        h, r, t = train_data
        paths = find_relation_context(h, r, t, entity_adj_table_with_rel)
        if r not in relation_adj_table:
            relation_adj_table[r] = paths
        else:
            relation_adj_table[r] += paths

    for k, v in relation_adj_table.items():
        relation_adj_table[k] = set([tuple(i) for i in v])

    if max_context_num_constraint:
        max_context_num = max_context
        for k, v in entity_adj_table.items():
            if len(v) > max_context_num:
                res = list(v)
                res = res[:max_context_num]
                entity_adj_table[k] = set(res)
        for k, v in relation_adj_table.items():
            if len(v) > max_context_num:
                res = list(v)
                res = res[:max_context_num]
                relation_adj_table[k] = set(res)
    else:
        max_context_num = 0
        for k, v in entity_adj_table.items():
            max_context_num = max(max_context_num, len(v))
        for k, v in relation_adj_table.items():
            max_context_num = max(max_context_num, len(v))

    entity_DAD = torch.Tensor(entity_total, max_context_num + 1, max_context_num + 1).cuda()
    relation_DAD = torch.Tensor(relation_total, max_context_num + 1, max_context_num + 1).cuda()

    for entity in range(entity_total):
        A = torch.eye(max_context_num + 1, max_context_num + 1).cuda()
        tmp = torch.ones(max_context_num + 1).cuda()
        A[0, :max_context_num + 1] = tmp
        A[:max_context_num + 1, 0] = tmp

        D = np.eye(max_context_num + 1, max_context_num + 1)
        i = list(range(max_context_num + 1))
        D[i, i] = 2
        D[0][0] = max_context_num + 1

        if entity in entity_adj_table:
            neighbours_list = list(entity_adj_table[entity])
            for index, neighbour in enumerate(neighbours_list):
                if neighbour not in entity_adj_table:
                    continue
                for index2, neighbour2 in enumerate(neighbours_list):
                    if index == index2:
                        continue
                    if neighbour2 in entity_adj_table[neighbour]:
                        A[index+1, index2+1] = 1
                        D[index+1][index+1] += 1

        D = np.linalg.inv(D)
        D = torch.Tensor(D).cuda()
        D[i, i] = torch.sqrt(D[i, i])

        entity_DAD[entity] = D.mm(A).mm(D)

    for relation in range(relation_total):
        A = torch.eye(max_context_num + 1, max_context_num + 1).cuda()
        tmp = torch.ones(max_context_num + 1).cuda()
        A[0, :max_context_num + 1] = tmp
        A[:max_context_num + 1, 0] = tmp

        D = np.eye(max_context_num + 1, max_context_num + 1)
        i = list(range(max_context_num + 1))
        D[i, i] = 2
        D[0][0] = max_context_num + 1

        if relation in relation_adj_table:
            neighbours_set = relation_adj_table[relation]
            for index, neighbour in enumerate(neighbours_set):
                if len(neighbour) != 1:
                    continue
                if neighbour[0] not in relation_adj_table:
                    continue
                adj_set = relation_adj_table[neighbour[0]]
                for index2, neighbour2 in enumerate(neighbours_set):
                    if index == index2:
                        continue
                    if neighbour2 in adj_set:
                        A[index+1, index2+1] = 1
                        D[index+1][index+1] += 1
        D = np.linalg.inv(D)
        D = torch.Tensor(D).cuda()
        i = list(range(max_context_num + 1))
        D[i, i] = torch.sqrt(D[i, i])

        relation_DAD[relation] = D.mm(A).mm(D)

    for k, v in entity_adj_table.items():
        res = list(v)
        entity_adj_table[k] = res + [entity_total] * (max_context_num - len(res))  # 补padding

    for k, v in relation_adj_table.items():
        res = []
        for i in v:
            if len(i) == 1:
                res.extend(list(i))
                res.append(relation_total)
            else:
                res.extend(list(i))

        relation_adj_table[k] = res + [relation_total] * 2 * (max_context_num - len(res) // 2)

    return entity_adj_table, relation_adj_table, max_context_num, entity_DAD, relation_DAD
    # return entity_adj_table, max_context_num, entity_DAD

print("Constructing adj table...")
entity_adj_table, relation_adj_table, max_context_num, entity_A, relation_A = construct_adj_table(train_list)
# entity_adj_table, relation_adj_table, max_context_num, entity_A, relation_A = dict(), dict(), 0, dict(), dict()
print("Constructing adj table completed.")


def bern_sampling_prepare(train_list):
    head2count = dict()
    tail2count = dict()
    for h, r, t in train_list:
        head2count[h] = head2count.get(h, 0) + 1
        tail2count[t] = tail2count.get(t, 0) + 1

    hpt = 0.0  # head per tail
    for t, count in tail2count.items():
        hpt += count
    hpt /= len(tail2count)

    tph = 0.0
    for h, count in head2count.items():
        tph += count
    tph /= len(head2count)

    return tph, hpt

def one_negative_sampling(golden_triple, train_set, tph=0.0, hpt=0.0):
    negative_triple = tuple()
    h, r, t = golden_triple

    if not bern:  # uniform sampling
        while True:
            e = random.randint(0, entity_total - 1)
            is_head = random.randint(0, 1)
            if is_head:
                if (e, r, t) in train_set:
                    continue
                else:
                    negative_triple = (e, r, t)
                    break
            else:
                if (h, r, e) in train_set:
                    continue
                else:
                    negative_triple = (h, r, e)
                    break
    else:
        sampling_head_prob = tph / (tph + hpt)

        while True:
            e = random.randint(0, entity_total - 1)
            is_head = random.random() > sampling_head_prob
            if is_head:
                if (e, r, t) in train_set:
                    continue
                else:
                    negative_triple = (e, r, t)
                    break
            else:
                if (h, r, e) in train_set:
                    continue
                else:
                    negative_triple = (h, r, e)
                    break

    return negative_triple

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
            negative_triples = one_negative_sampling(golden_triple, train_set, tph, hpt)
            nhs[j][i], nrs[j][i], nts[j][i] = negative_triples

    return torch.IntTensor(phs).cuda(), torch.IntTensor(prs).cuda(), torch.IntTensor(pts).cuda(), torch.IntTensor(
        nhs).cuda(), torch.IntTensor(nrs).cuda(), torch.IntTensor(nts).cuda()


def get_batch(batch, epoch, phs, prs, pts, nhs, nrs, nts):
    r = min((batch + 1) * batch_size, len(phs))

    return (phs[batch * batch_size: r], prs[batch * batch_size: r], pts[batch * batch_size: r]), \
           (nhs[epoch, batch * batch_size: r], nrs[epoch, batch * batch_size: r], nts[epoch, batch * batch_size: r])


def get_batch_A(triples):
    h, r, t = triples
    return entity_A[h.cpu().numpy()], relation_A[r.cpu().numpy()], entity_A[t.cpu().numpy()]


def get_head_batch(golden_triple):
    head_batch = np.zeros((entity_total, 3), dtype=np.int32)
    head_batch[:, 0] = np.array(list(range(entity_total)))
    head_batch[:, 1] = np.array([golden_triple[1]] * entity_total)
    head_batch[:, 2] = np.array([golden_triple[2]] * entity_total)
    return head_batch


def get_tail_batch(golden_triple):
    tail_batch = np.zeros((entity_total, 3), dtype=np.int32)
    tail_batch[:, 0] = np.array([golden_triple[0]] * entity_total)
    tail_batch[:, 1] = np.array([golden_triple[1]] * entity_total)
    tail_batch[:, 2] = np.array(list(range(entity_total)))
    return tail_batch


def load_o_emb(epoch, input=False, mode='train'):
    if input:
        with open('./res/entity_o_parameters%s' % str(epoch), "r") as f:
            emb = json.loads(f.read())
            # entity_emb = torch.Tensor(len(emb), dim)
            entity_emb = torch.Tensor(entity_total, dim)
            for k, v in emb.items():
                entity_emb[int(k)] = torch.Tensor(v)

        with open('./res/relation_o_parameters%s' % str(epoch), "r") as f:
            emb = json.loads(f.read())
            # relation_emb = torch.Tensor(len(emb), dim)
            relation_emb = torch.Tensor(relation_total, dim)
            for k, v in emb.items():
                relation_emb[int(k)] = torch.Tensor(v)
        return entity_emb.cuda(), relation_emb.cuda()

    else:
        if mode == 'online':
            with open(res_dir + 'online/entity_o_parameters' + str(epoch), "r") as f:
                ent_emb = json.loads(f.read())
            with open(res_dir + 'online/relation_o_parameters' + str(epoch), "r") as f:
                rel_emb = json.loads(f.read())
        else:
            with open(res_dir + 'entity_o_parameters' + str(epoch), "r") as f:
                ent_emb = json.loads(f.read())
            with open(res_dir + 'relation_o_parameters' + str(epoch), "r") as f:
                rel_emb = json.loads(f.read())

        # entity_emb = torch.Tensor(len(ent_emb), dim)
        entity_emb = torch.Tensor(entity_total, dim)
        for k, v in ent_emb.items():
            entity_emb[int(k)] = torch.Tensor(v)

        # relation_emb = torch.Tensor(len(rel_emb), dim)
        relation_emb = torch.Tensor(relation_total, dim)
        for k, v in rel_emb.items():
            relation_emb[int(k)] = torch.Tensor(v)

        return entity_emb.cuda(), relation_emb.cuda()


def load_parameters(epoch):
    with open('./res/' + 'all_parameters' + str(epoch), 'r') as f:
        emb = json.loads(f.read())
        entity_emb = emb['entity_emb.weight']
        relation_emb = emb['relation_emb.weight']
        entity_context = emb['entity_context.weight']
        relation_context = emb['relation_context.weight']
        entity_gcn_weight = emb['entity_gcn_weight']
        relation_gcn_weight = emb['relation_gcn_weight']
        gate_entity = emb['gate_entity']
        gate_relation = emb['gate_relation']
        v_ent = emb['v_ent']
        v_rel = emb['v_rel']

        # give index, return embedding
        return torch.Tensor(entity_emb).cuda(), torch.Tensor(relation_emb).cuda(), \
               torch.Tensor(entity_context).cuda(), torch.Tensor(relation_context).cuda(), \
               torch.Tensor(entity_gcn_weight).cuda(), torch.Tensor(relation_gcn_weight).cuda(), \
               torch.Tensor(gate_entity).cuda(), torch.Tensor(gate_relation).cuda(), \
               torch.Tensor(v_ent).cuda(), torch.Tensor(v_rel).cuda()


def get_transe_embdding(input=True):
    with open('./res/' + transe_model_file, "r") as f:
        emb = json.loads(f.read())
        entity_emb = emb['ent_embeddings.weight']
        relation_emb = emb['rel_embeddings.weight']
        # entity_emb = emb['entity_emb']
        # relation_emb = emb['relation_emb']
        if input:
            return torch.Tensor(entity_emb).cuda(), torch.Tensor(relation_emb).cuda()
        else:
            return entity_emb, relation_emb


def get_affected_entities(entity_set1, entity_set2, entity_context_dict1, entity_context_dict2):
    affected_entities_set1 = set()
    for entity in entity_set2 - entity_set1:
        for target_entity, context_set in entity_context_dict2.items():
            if entity in context_set:
                affected_entities_set1.add(target_entity)

    affected_entities_set1 = affected_entities_set1 & entity_set1

    affected_entities_set2 = set()
    for entity in entity_set1 - entity_set2:
        for target_entity, context_set in entity_context_dict1.items():
            if entity in context_set:
                affected_entities_set2.add(target_entity)
    affected_entities_set2 = affected_entities_set2 & entity_set2

    return affected_entities_set1 | affected_entities_set2


def get_affected_relations(snapshot2_train_set, relation_set1, relation_set2):
    relation_dict = dict()
    head_entity_dict = dict()
    relation_context_dict = dict()

    for (h, r, t) in snapshot2_train_set:
        head_entity_dict.setdefault(h, list()).append([r, t])
        relation_dict.setdefault(r, list()).append([h, t])

    for relation, nodes_set in relation_dict.items():
        relation_context_dict[relation] = set()
        for nodes in nodes_set:
            head, tail = nodes[:2]
            for edge in head_entity_dict[head]:
                if edge[1] == tail:
                    relation_context_dict[relation].add(edge[0])
            relation_context_dict[relation].remove(relation)

            for edge1 in head_entity_dict[head]:
                if edge1[1] in head_entity_dict:
                    for edge2 in head_entity_dict[edge1[1]]:
                        if edge2[1] == tail:
                            relation_context_dict[relation].add("%s+%s" % (edge1[0], edge2[0]))

    affected_relations_set = set()
    for new_relation in (relation_set2 - relation_set1):
        for relation, contexts_set in relation_context_dict.items():
            for context_relation in contexts_set:
                if new_relation in context_relation:
                    affected_relations_set.add(relation)
                    break

    affected_relations_set = affected_relations_set - (relation_set2 - relation_set1)
    return affected_relations_set


def analyse_affected_data():
    snapshot1_train_list = read_file(file_name='./data/' + dataset_v1 + '/train2id.txt')
    snapshot2_train_list = read_file(file_name='./data/' + dataset_v2 + '/train2id.txt')

    snapshot1_train_set = set(convert_id_to_text(snapshot1_train_list, dataset_v1))
    snapshot2_train_set = set(convert_id_to_text(snapshot2_train_list, dataset_v2))

    entity_set1, relation_set1, entity_context_dict1 = get_basic_info(snapshot1_train_set)
    entity_set2, relation_set2, entity_context_dict2 = get_basic_info(snapshot2_train_set)

    affected_entities = get_affected_entities(entity_set1, entity_set2, entity_context_dict1, entity_context_dict2)
    affected_relations = get_affected_relations(snapshot2_train_set, relation_set1, relation_set2)

    new_entities = entity_set2 - entity_set1
    new_relations = relation_set2 - relation_set1

    affected_triples = list()
    for (h, r, t) in snapshot2_train_set:
        if h in affected_entities or h in new_entities or t in affected_entities or t in new_entities or \
                r in affected_relations or r in new_relations:
            affected_triples.append((h, r, t))

    return affected_entities, affected_relations, affected_triples


def construct_snapshots_mapping_dict():
    snapshot1_entity2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v1 + '/entity2id.txt')
    snapshot1_relation2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v1 + '/relation2id.txt')

    snapshot2_entity2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v2 + '/entity2id.txt')
    snapshot2_relation2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v2 + '/relation2id.txt')

    entity_mapping_dict = dict()
    for entity, id in snapshot1_entity2id_dict.items():
        if entity in snapshot2_entity2id_dict:
            entity_mapping_dict[id] = snapshot2_entity2id_dict[entity]

    relation_mapping_dict = dict()
    for relation, id in snapshot1_relation2id_dict.items():
        if relation in snapshot2_relation2id_dict:
            relation_mapping_dict[id] = snapshot2_relation2id_dict[relation]

    return snapshot1_entity2id_dict, snapshot1_relation2id_dict, snapshot2_entity2id_dict, snapshot2_relation2id_dict, \
        entity_mapping_dict, relation_mapping_dict


def analyse_snapshots():
    snapshot1_entity2id_dict, snapshot1_relation2id_dict, snapshot2_entity2id_dict, snapshot2_relation2id_dict, \
     entity_mapping_dict, relation_mapping_dict = construct_snapshots_mapping_dict()

    added_entities = set()
    for entity, id in snapshot2_entity2id_dict.items():
        if entity not in snapshot1_entity2id_dict:
            added_entities.add(id)

    added_relations = set()
    for relation, id in snapshot2_relation2id_dict.items():
        if relation not in snapshot1_relation2id_dict:
            added_relations.add(id)

    affected_entities, affected_relations, affected_triples = analyse_affected_data()

    affected_entities = set([snapshot2_entity2id_dict[e] for e in affected_entities])
    affected_relations = set([snapshot2_relation2id_dict[r] for r in affected_relations])
    affected_triples = [(snapshot2_entity2id_dict[h], snapshot2_relation2id_dict[r], snapshot2_entity2id_dict[t]) for (h, r, t) in affected_triples]

    print("affected entities:%d" % len(affected_entities))
    print("affected relations:%d" % len(affected_relations))
    print("affected triples:%d" % len(affected_triples))
    print("added entities:%d" % len(added_entities))
    print("addded relations:%d" % len(added_relations))

    return affected_entities, affected_relations, affected_triples, added_entities, added_relations, entity_mapping_dict, relation_mapping_dict


def prepare_online_data(embedding_model_file):
    print("Analysing snapshots...")
    affected_entities, affected_relations, \
    affected_triples, \
    added_entities, added_relations, \
    entity_mapping_dict, relation_mapping_dict = analyse_snapshots()
    print("Analyse snapshots completed.")

    print("Negatvie sampling...")
    phs = np.zeros(len(affected_triples), dtype=int)
    prs = np.zeros(len(affected_triples), dtype=int)
    pts = np.zeros(len(affected_triples), dtype=int)
    nhs = np.zeros((train_times, len(affected_triples)), dtype=int)
    nrs = np.zeros((train_times, len(affected_triples)), dtype=int)
    nts = np.zeros((train_times, len(affected_triples)), dtype=int)

    tph, hpt = bern_sampling_prepare(train_list)
    train_set = set(train_list)
    for i, golden_triple in enumerate(affected_triples):
        # print(i, end='\r')
        phs[i], prs[i], pts[i] = golden_triple

        for j in range(train_times):
            negative_triple = one_negative_sampling(golden_triple, train_set, tph, hpt)
            nhs[j][i], nrs[j][i], nts[j][i] = negative_triple

    phs = torch.LongTensor(phs).cuda()
    prs = torch.LongTensor(prs).cuda()
    pts = torch.LongTensor(pts).cuda()
    nhs = torch.LongTensor(nhs).cuda()
    nrs = torch.LongTensor(nrs).cuda()
    nts = torch.LongTensor(nts).cuda()
    print("Negatvie sampling finished.")

    entity_emb, relation_emb, entity_context, relation_context, \
    entity_gcn_weight, relation_gcn_weight, gate_entity, gate_relation, \
    v_entity, v_relation = load_parameters(embedding_model_file)

    new_entity_emb = torch.zeros(entity_total, dim).cuda()
    nn.init.xavier_uniform_(new_entity_emb)
    for id1, id2 in entity_mapping_dict.items():
        new_entity_emb[id2] = entity_emb[id1]

    new_relation_emb = torch.zeros(relation_total, dim).cuda()
    nn.init.xavier_uniform_(new_relation_emb)
    for id1, id2 in relation_mapping_dict.items():
        new_relation_emb[id2] = relation_emb[id1]

    new_entity_context = torch.zeros(entity_total + 1, dim).cuda()
    nn.init.xavier_uniform_(new_entity_context)
    for id1, id2 in entity_mapping_dict.items():
        new_entity_context[id2] = entity_context[id1]

    new_relation_context = torch.zeros(relation_total + 1, dim).cuda()
    nn.init.xavier_uniform_(new_relation_context)
    for id1, id2 in relation_mapping_dict.items():
        new_relation_context[id2] = relation_context[id1]

    entity_o_emb, relation_o_emb = load_o_emb(embedding_model_file, input=True)

    new_entity_o_emb = torch.zeros(entity_total, dim).cuda()
    nn.init.xavier_uniform_(new_entity_o_emb)
    for id1, id2 in entity_mapping_dict.items():
        new_entity_o_emb[id2] = entity_o_emb[id1]

    new_relation_o_emb = torch.zeros(relation_total, dim).cuda()
    nn.init.xavier_uniform_(new_relation_o_emb)
    for id1, id2 in relation_mapping_dict.items():
        new_relation_o_emb[id2] = relation_o_emb[id1]

    return new_entity_emb, new_relation_emb, new_entity_context, new_relation_context, \
           entity_gcn_weight, relation_gcn_weight, gate_entity, gate_relation, v_entity, v_relation, \
           phs, prs, pts, nhs, nrs, nts, affected_entities, affected_relations, added_entities, added_relations, \
           new_entity_o_emb, new_relation_o_emb



